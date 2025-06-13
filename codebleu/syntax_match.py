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
from .utils import get_tree_sitter_language, cur_node_contains_mod_line, get_mod_lines

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
        ref_diff_item = ref_diffs[i]
        cand_diff = cand_diffs[i]

        for j, reference in enumerate(references_sample):
            # Select the appropriate diff for this particular reference.
            if isinstance(ref_diff_item, list):
                if j < len(ref_diff_item):
                    ref_diff = ref_diff_item[j]
                else:
                    # Fallback to using the first diff if not enough provided.
                    ref_diff = ref_diff_item[0]
            else:
                ref_diff = ref_diff_item

            # Clean code snippets
            try:
                cleaned_candidate = remove_comments_and_docstrings(candidate, lang)
            except Exception:
                cleaned_candidate = candidate

            try:
                cleaned_reference = remove_comments_and_docstrings(reference, lang)
            except Exception:
                cleaned_reference = reference

            candidate_tree = parser.parse(bytes(cleaned_candidate, "utf8")).root_node
            reference_tree = parser.parse(bytes(cleaned_reference, "utf8")).root_node

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
