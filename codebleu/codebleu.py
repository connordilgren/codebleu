# Copyright (c) Microsoft Corporation.
# Copyright (c) 2023 Konstantin Chernyshev.
# Licensed under the MIT license.
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from . import bleu, dataflow_match, syntax_match, weighted_ngram_match
from .utils import AVAILABLE_LANGS, get_tree_sitter_language

PACKAGE_DIR = Path(__file__).parent


def calc_codebleu(
    references: Union[List[str], List[List[str]]],
    predictions: List[str],
    dev_diffs: Union[List[str], List[List[str]]],
    llm_diffs: List[str],
    lang: str,
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    tokenizer: Optional[Callable] = None,
    keywords_dir: Path = PACKAGE_DIR / "keywords",
) -> Dict[str, float]:
    """Calculate CodeBLEU score

    Args:
        references: list of lists with references (outer list is N examples, each inner list has M correct answers). This is the base code with the dev diffs applied.
        predictions: list of predictions (N examples). This is the base code with the llm diffs applied.
        dev_diffs: list of lists with reference diffs (outer list is N examples, each inner list has M correct answers)
        llm_diffs: list of prediction diffs (N examples)
        lang: input language, one of AVAILABLE_LANGS
        weights: weights of the ngram_match, weighted_ngram_match, syntax_match, and dataflow_match respectively
        tokenizer: tokenizer function, Defaults to lambda s: s.split()
        keywords_dir: path to the directory with keywords files
        lang_so_file: path to the .so file with the parser for the language

    Return:
        Scores dict
    """
    assert len(references) == len(predictions), "Number of references and predictions should be the same"
    assert len(dev_diffs) == len(llm_diffs), "Number of dev diffs and llm diffs should be the same"
    assert len(references) == len(dev_diffs), "Number of references and dev diffs should be the same"
    assert lang in AVAILABLE_LANGS, f"Language {lang} is not supported (yet). Available languages: {AVAILABLE_LANGS}"
    assert len(weights) == 4, "weights should be a tuple of 4 floats (alpha, beta, gamma, theta)"
    assert keywords_dir.exists(), f"keywords_dir {keywords_dir} does not exist"

    # get the tree-sitter language for a given language
    tree_sitter_language = get_tree_sitter_language(lang)

    # preprocess inputs
    references = [[x.strip() for x in ref] if isinstance(ref, list) else [ref.strip()] for ref in references]
    hypothesis = [x.strip() for x in predictions]

    # calculate ngram match (BLEU)
    if tokenizer is None:

        def tokenizer(s):
            return s.split()

    tokenized_hyps = [tokenizer(x) for x in hypothesis]
    tokenized_refs = [[tokenizer(x) for x in reference] for reference in references]

    # ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)
    ngram_match_score = 0  # TODO

    # # calculate weighted ngram match
    # with open(keywords_dir / (lang + ".txt"), "r", encoding="utf-8") as f:
    #     keywords = [x.strip() for x in f.readlines()]

    # def make_weights(reference_tokens, key_word_list):
    #     return {token: 1 if token in key_word_list else 0.2 for token in reference_tokens}

    # tokenized_refs_with_weights = [
    #     [[reference_tokens, make_weights(reference_tokens, keywords)] for reference_tokens in reference]
    #     for reference in tokenized_refs
    # ]

    # weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)
    weighted_ngram_match_score = 0  # TODO

    # calculate syntax match
    add_syntax_match_score = syntax_match.corpus_syntax_match(
        references, hypothesis, dev_diffs, llm_diffs, lang, tree_sitter_language=tree_sitter_language, view='added'
    )

    del_syntax_match_score = syntax_match.corpus_syntax_match(
        references, hypothesis, dev_diffs, llm_diffs, lang, tree_sitter_language=tree_sitter_language, view='deleted'
    )

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(
        references, hypothesis, lang, tree_sitter_language=tree_sitter_language
    )

    alpha, beta, gamma, theta = weights
    add_code_bleu_score = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * add_syntax_match_score
        + theta * (dataflow_match_score or 1)
    )

    del_code_bleu_score = (
        alpha * ngram_match_score
        + beta * weighted_ngram_match_score
        + gamma * del_syntax_match_score
        + theta * (dataflow_match_score or 1)
    )

    code_bleu_score = (add_code_bleu_score + del_code_bleu_score) / 2

    return {
        "codebleu": code_bleu_score,

        "add_codebleu": add_code_bleu_score,
        "del_codebleu": del_code_bleu_score,

        "add_ngram_match_score": ngram_match_score,
        "del_ngram_match_score": ngram_match_score,

        "add_weighted_ngram_match_score": weighted_ngram_match_score,
        "del_weighted_ngram_match_score": weighted_ngram_match_score,

        "add_syntax_match_score": add_syntax_match_score,
        "del_syntax_match_score": del_syntax_match_score,

        "add_dataflow_match_score": dataflow_match_score,
        "del_dataflow_match_score": dataflow_match_score,
    }


if __name__ == "__main__":
    # note: for multi-file, we'd need to split the diffs into multiple diffs for each file
    # each reference would be a singular list of the dev-applied diff for each file
    # each prediction would be the llm-applied diff for each file

    dev_code_path = "examples/dev_packet-cdma2k.c"
    llm_code_path = "examples/llm_packet-cdma2k.c"
    dev_diff_path = "examples/11633_dev.diff"
    llm_diff_path = "examples/11633_llm.diff"

    with open(dev_code_path, "r", encoding="utf-8") as f:
        dev_code = f.read()

    with open(llm_code_path, "r", encoding="utf-8") as f:
        llm_code = f.read()

    with open(dev_diff_path, "r", encoding="utf-8") as f:
        dev_diff = f.read()

    with open(llm_diff_path, "r", encoding="utf-8") as f:
        llm_diff = f.read()

    scores = calc_codebleu(
        references=[dev_code], 
        predictions=[llm_code], 
        dev_diffs=[dev_diff],
        llm_diffs=[llm_diff],
        lang="c",
        weights=(0.25, 0.25, 0.25, 0.25),
        tokenizer=None,
        keywords_dir=PACKAGE_DIR / "keywords",
    )

    print(scores)
