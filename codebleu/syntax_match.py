# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tree_sitter import Parser

from .parser import (
    DFG_csharp,
    DFG_go,
    DFG_java,
    DFG_javascript,
    DFG_php,
    DFG_python,
    DFG_ruby,
    remove_comments_and_docstrings,
)
from .utils import get_tree_sitter_language

dfg_function = {
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
    "c_sharp": DFG_csharp,
}


def calc_syntax_match(references, candidate, lang):
    return corpus_syntax_match([references], [candidate], lang)


def cur_node_contains_mod_line(cur_node, mod_lines: list[int]) -> bool:
    """
    returns true if the cur_node contains any of the mod_lines

    Args:
        cur_node: the current node to check
        mod_lines: the lines that are modified in the candidate code
    """
    cur_node_start_row = cur_node.start_point[0] + 1 # first line is 0
    cur_node_end_row = cur_node.end_point[0] + 1

    return any(cur_node_start_row <= mod_line <= cur_node_end_row for mod_line in mod_lines)


import re

def get_mod_lines(diff, view):
    """
    Return the 1-based line numbers that are *added* or *deleted* in a
    unified-diff string.

    Args:
        diff (str): Unified diff text containing exactly one file's changes.
        view (str): Either 'added' (report new-file line numbers for lines
                    beginning with '+') or 'deleted' (report old-file line
                    numbers for lines beginning with '-').

    Returns:
        List[int]: Sorted list of line numbers matching the requested view.

    Raises:
        ValueError: If *view* is not 'added' or 'deleted'.
    """
    if view not in {"added", "deleted"}:
        raise ValueError("view must be 'added' or 'deleted'")

    # @@ -a,b +c,d @@  (b or d may be omitted ⇒ 1)
    hunk_pat = re.compile(r'^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@')
    mod_lines = []

    old_ln = new_ln = None  # current line numbers for old/new files

    for line in diff.splitlines():
        # Detect the start of a new hunk and reset counters.
        m = hunk_pat.match(line)
        if m:
            old_ln = int(m.group(1))
            new_ln = int(m.group(3))
            continue

        # Ignore everything until the first hunk header.
        if old_ln is None:
            continue

        # Classify the line and update counters.
        if line.startswith(' '):          # context line
            old_ln += 1
            new_ln += 1
        elif line.startswith('-'):        # deletion
            if view == "deleted":
                mod_lines.append(old_ln)
            old_ln += 1
        elif line.startswith('+'):        # addition
            if view == "added":
                mod_lines.append(new_ln)
            new_ln += 1
        # Other diff metadata (e.g., "\ No newline…") is ignored.

    return mod_lines


def corpus_syntax_match(references, candidates, ref_diffs, cand_diffs, lang, tree_sitter_language=None, view='added'):
    """
    Args:
        view: 'added' or 'deleted', for which lines to consider from the diff
    """

    if not tree_sitter_language:
        tree_sitter_language = get_tree_sitter_language(lang)

    parser = Parser()
    parser.language = tree_sitter_language
    match_count = 0
    match_count_candidate_to_reference = 0
    total_count = 0

    for i in range(len(candidates)):
        references_sample = references[i]
        candidate = candidates[i]
        ref_diff = ref_diffs[i]
        cand_diff = cand_diffs[i]
        for reference in references_sample:
            try:
                candidate = remove_comments_and_docstrings(candidate, lang)
            except Exception:
                pass
            try:
                reference = remove_comments_and_docstrings(reference, lang)
            except Exception:
                pass

            candidate_tree = parser.parse(bytes(candidate, "utf8")).root_node

            reference_tree = parser.parse(bytes(reference, "utf8")).root_node

            def get_all_sub_trees(root_node):
                node_stack = []
                sub_tree_sexp_list = []
                depth = 1
                node_stack.append([root_node, depth])
                while len(node_stack) != 0:
                    cur_node, cur_depth = node_stack.pop()
                    sub_tree_sexp_list.append([str(cur_node), cur_depth])
                    for child_node in cur_node.children:
                        if len(child_node.children) != 0:
                            depth = cur_depth + 1
                            node_stack.append([child_node, depth])
                return sub_tree_sexp_list

            def get_all_sub_trees_in_diff(root_node, mod_lines):
                node_stack = []
                sub_tree_sexp_list = []
                depth = 1
                node_stack.append([root_node, depth])
                while len(node_stack) != 0:
                    cur_node, cur_depth = node_stack.pop()
                    if cur_node_contains_mod_line(cur_node, mod_lines):
                        sub_tree_sexp_list.append([str(cur_node), cur_depth])
                    for child_node in cur_node.children:
                        if len(child_node.children) != 0:
                            depth = cur_depth + 1
                            node_stack.append([child_node, depth])
                return sub_tree_sexp_list

            cand_mod_lines = get_mod_lines(cand_diff, view)
            ref_mod_lines = get_mod_lines(ref_diff, view)

            cand_sexps = [x[0] for x in get_all_sub_trees_in_diff(candidate_tree, cand_mod_lines)]
            ref_sexps = [x[0] for x in get_all_sub_trees_in_diff(reference_tree, ref_mod_lines)]

            # TODO: fix, now we count number of reference subtrees matching candidate,
            #       but we should count number of candidate subtrees matching reference
            #       See (4) in "3.2 Syntactic AST Match" of https://arxiv.org/pdf/2009.10297.pdf
            for sub_tree in ref_sexps:
                if sub_tree in cand_sexps:
                    match_count += 1

            for sub_tree in cand_sexps:
                if sub_tree in ref_sexps:
                    match_count_candidate_to_reference += 1

            total_count += len(ref_sexps)
    # print(f'match_count       {match_count} / {total_count}')
    # print(f'match_count_fixed {match_count_candidate_to_reference} / {total_count}')
    score = match_count / total_count if total_count > 0 else 0  # total_count is 0 if tree-sitter fails to parse the code
    return score
