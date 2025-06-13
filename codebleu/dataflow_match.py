# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from typing import Optional

from tree_sitter import Parser

from .parser import (
    DFG_csharp,
    DFG_go,
    DFG_java,
    DFG_javascript,
    DFG_php,
    DFG_python,
    DFG_ruby,
    DFG_rust,
    index_to_code_token,
    remove_comments_and_docstrings,
    tree_to_token_index,
)
from .utils import get_tree_sitter_language, get_mod_lines

dfg_function = {
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
    "c_sharp": DFG_csharp,
    "c": DFG_csharp,  # XLCoST uses C# parser for C
    "cpp": DFG_csharp,  # XLCoST uses C# parser for C++
    "rust": DFG_rust,
}


def calc_dataflow_match(
    references,
    candidate,
    lang,
    langso_so_file,
    ref_diff: Optional[str] = None,
    cand_diff: Optional[str] = None,
    view: str = "added",
):
    """Compute the data-flow match between *candidate* and *references*.

    The optional *ref_diff* and *cand_diff* arguments allow restricting the
    comparison to only those data-flow nodes that originate from lines that are
    part of the respective unified-diffs. Set *view* to "added" (default) to
    consider added lines or to "deleted" to consider removed lines.
    """

    ref_diffs = [ref_diff] if ref_diff is not None else None
    cand_diffs = [cand_diff] if cand_diff is not None else None

    return corpus_dataflow_match(
        [references],
        [candidate],
        lang,
        ref_diffs=ref_diffs,
        cand_diffs=cand_diffs,
        tree_sitter_language=langso_so_file,
        view=view,
    )


# NOTE: *ref_diffs* and *cand_diffs* are lists of unified diff strings whose
#       indices correspond to *references* and *candidates* respectively. If
#       they are *None*, the behaviour is identical to the original
#       implementation (i.e., the entire file is considered).
#       The *view* parameter can be either 'added' or 'deleted' and determines
#       whether the diff lines refer to additions or deletions.


def corpus_dataflow_match(
    references,
    candidates,
    lang,
    ref_diffs=None,
    cand_diffs=None,
    tree_sitter_language=None,
    view: str = "added",
):
    if not tree_sitter_language:
        tree_sitter_language = get_tree_sitter_language(lang)

    parser = Parser()
    parser.language = tree_sitter_language
    parser = [parser, dfg_function[lang]]
    match_count = 0
    total_count = 0

    for i in range(len(candidates)):
        references_sample = references[i]
        candidate_code = candidates[i]

        # Compute candidate diff line numbers if diffs are provided
        cand_mod_lines = []
        if cand_diffs is not None and i < len(cand_diffs):
            cand_mod_lines = get_mod_lines(cand_diffs[i], view)

        # Pre-compute candidate DFG and mapping once per reference set
        try:
            cleaned_candidate_code = remove_comments_and_docstrings(candidate_code, lang)
        except Exception:
            cleaned_candidate_code = candidate_code

        cand_dfg_full = get_data_flow(cleaned_candidate_code, parser)
        cand_idx_to_line = _idx_to_line_map(cleaned_candidate_code, parser)

        # Filter candidate DFG if diff information is available
        cand_dfg = _filter_dfg_by_lines(cand_dfg_full, cand_idx_to_line, cand_mod_lines)

        # Normalise ONLY ONCE for efficiency. We'll copy later when we remove matched items.
        normalized_cand_dfg = normalize_dataflow(list(cand_dfg))

        for j, reference_code in enumerate(references_sample):
            # Compute reference diff lines if provided
            ref_mod_lines = []
            if ref_diffs is not None and i < len(ref_diffs):
                # ref_diffs is expected to be list of list? We'll assume list aligned to references list.
                ref_diff_item = ref_diffs[i]
                # If there are multiple references per sample, ref_diffs[i] could be list of diff strings.
                # Support both structures.
                if isinstance(ref_diff_item, list):
                    if j < len(ref_diff_item):
                        ref_mod_lines = get_mod_lines(ref_diff_item[j], view)
                else:
                    # Single diff for all refs in sample
                    ref_mod_lines = get_mod_lines(ref_diff_item, view)

            try:
                cleaned_ref_code = remove_comments_and_docstrings(reference_code, lang)
            except Exception:
                cleaned_ref_code = reference_code

            ref_dfg_full = get_data_flow(cleaned_ref_code, parser)
            ref_idx_to_line = _idx_to_line_map(cleaned_ref_code, parser)
            ref_dfg = _filter_dfg_by_lines(ref_dfg_full, ref_idx_to_line, ref_mod_lines)

            normalized_ref_dfg = normalize_dataflow(ref_dfg)

            if len(normalized_ref_dfg) > 0:
                total_count += len(normalized_ref_dfg)
                # Work on a copy of normalized_cand_dfg to prevent removing from original.
                cand_dfg_working = normalized_cand_dfg.copy()
                for dataflow in normalized_ref_dfg:
                    if dataflow in cand_dfg_working:
                        match_count += 1
                        cand_dfg_working.remove(dataflow)
    if total_count == 0:
        logging.warning(
            "WARNING: There is no reference data-flows extracted from the whole corpus, "
            "and the data-flow match score degenerates to 0. Please consider ignoring this score."
        )
        return 0
    score = match_count / total_count
    return score


def get_data_flow(code, parser):
    try:
        tree = parser[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split("\n")
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except Exception:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except Exception:
        code.split()
        dfg = []
    # merge nodes
    dic = {}
    for d in dfg:
        if d[1] not in dic:
            dic[d[1]] = d
        else:
            dic[d[1]] = (
                d[0],
                d[1],
                d[2],
                list(set(dic[d[1]][3] + d[3])),
                list(set(dic[d[1]][4] + d[4])),
            )
    DFG = []
    for d in dic:
        DFG.append(dic[d])
    dfg = DFG
    return dfg


def normalize_dataflow_item(dataflow_item):
    var_name = dataflow_item[0]
    dataflow_item[1]
    relationship = dataflow_item[2]
    par_vars_name_list = dataflow_item[3]
    dataflow_item[4]

    var_names = list(set(par_vars_name_list + [var_name]))
    norm_names = {}
    for i in range(len(var_names)):
        norm_names[var_names[i]] = "var_" + str(i)

    norm_var_name = norm_names[var_name]
    relationship = dataflow_item[2]
    norm_par_vars_name_list = [norm_names[x] for x in par_vars_name_list]

    return (norm_var_name, relationship, norm_par_vars_name_list)


def normalize_dataflow(dataflow):
    var_dict = {}
    i = 0
    normalized_dataflow = []
    for item in dataflow:
        var_name = item[0]
        relationship = item[2]
        par_vars_name_list = item[3]
        for name in par_vars_name_list:
            if name not in var_dict:
                var_dict[name] = "var_" + str(i)
                i += 1
        if var_name not in var_dict:
            var_dict[var_name] = "var_" + str(i)
            i += 1
        normalized_dataflow.append(
            (
                var_dict[var_name],
                relationship,
                [var_dict[x] for x in par_vars_name_list],
            )
        )
    return normalized_dataflow


# Helper function to build mapping from token idx to source line number
def _idx_to_line_map(code: str, parser):
    """Return a mapping from the token enumeration index produced in get_data_flow
    to the 1-based source line number of that token.

    This reproduces the token enumeration logic used inside `get_data_flow` in order
    to discover which data-flow nodes originate from lines modified in a diff.
    The logic intentionally mirrors the implementation found in `get_data_flow`
    so that the indices are aligned.
    """
    try:
        tree = parser[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        idx_to_line: dict[int, int] = {}
        for idx, index in enumerate(tokens_index):
            # `index` is a tuple ((row, col), (row, col)) – use the starting row.
            start_line = index[0][0] + 1  # convert 0-based to 1-based
            idx_to_line[idx] = start_line
        return idx_to_line
    except Exception:
        # When tree-sitter fails to parse.
        return {}


# Helper to keep only those DFG items that are related to the supplied line numbers
def _filter_dfg_by_lines(dfg: list, idx_to_line: dict[int, int], mod_lines: list[int]):
    """Filter a data-flow graph so that only items whose variable or any of its
    parent variables are defined on one of *mod_lines* are kept.
    """
    mod_lines_set = set(mod_lines)
    if not mod_lines_set:
        return dfg  # nothing to filter

    filtered = []
    for item in dfg:
        if len(item) < 5:
            # Unexpected shape – keep for safety.
            continue
        var_idx = item[1]
        parent_indices = item[4]

        line_numbers = []
        if var_idx in idx_to_line:
            line_numbers.append(idx_to_line[var_idx])
        for pidx in parent_indices:
            if pidx in idx_to_line:
                line_numbers.append(idx_to_line[pidx])

        if any(ln in mod_lines_set for ln in line_numbers):
            filtered.append(item)

    return filtered
