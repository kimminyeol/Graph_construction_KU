"""Offline tests for the abstract check (no API, no GPU): the decision rules of kg/steps/verify_abstracts.py."""
from __future__ import annotations

from kg.steps.verify_abstracts import DEFAULTS, decide, is_latin, lexical_overlap, title_in_abstract


def row(title, abstract, cos, reg=None, reg_cos=None):
    return {"title": title, "abstract": abstract, "cos": cos, "reg_abstract": reg, "reg_cos": reg_cos, "reg_source": "arxiv"}


GOOD = "We study how to align language models with human preferences using a simple classification loss instead of reinforcement learning, and show it is stable."
FOREIGN = "This document summarizes the results of a rigorous scientific verification of over 130 hypotheses underlying the TRIAD architecture using a consensus tool."
BLURB = "Inderscience is a global company, a dynamic leading independent journal publisher disseminates the latest research across the broad fields of science."


def test_registry_mismatch_is_replaced_only_when_the_registry_abstract_fits_the_title_better():
    a, b, c = decide([
        row("Direct Preference Optimization: Your Language Model is Secretly a Reward Model", FOREIGN, 0.55, reg=GOOD, reg_cos=0.80),   # foreign abstract, registry fits
        row("Expert finding in legal community question answering", "A reproducibility study of expert finding for legal questions on community platforms.", 0.86, reg=GOOD, reg_cos=0.86),   # different version of the abstract: left alone
        row("Direct Preference Optimization: Your Language Model is Secretly a Reward Model", GOOD, 0.80, reg=GOOD, reg_cos=0.80),     # identical: ok
    ], DEFAULTS)
    assert a["action"] == "replaced" and a["new_abstract"] == GOOD and a["reason"] == "registry_mismatch"
    assert b["action"] == "ok" and c["action"] == "ok"


def test_duplicate_abstract_under_different_titles_is_suspect_but_a_shared_title_is_not():
    rows = [row(f"Paper about topic number {i} in robotics", BLURB, 0.40) for i in range(3)] + [row("Same title twice", GOOD, 0.8), row("Same title twice", GOOD, 0.8)]
    out = decide(rows, DEFAULTS)
    assert [d["action"] for d in out[:3]] == ["suspect"] * 3 and out[0]["reason"] == "duplicate_abstract"
    assert [d["action"] for d in out[3:]] == ["ok", "ok"]                              # two versions of one paper share an abstract legitimately
    versions = decide([row("VioLA: Conditional Language Models for Speech", GOOD, 0.77), row("VioLA: Unified Codec Language Models for Speech", GOOD, 0.79)], DEFAULTS)
    assert [d["action"] for d in versions] == ["ok", "ok"]                              # different titles but a correct abstract (high similarity): not a blurb
    assert decide([row("Title Pending 10324", FOREIGN, 0.46)], DEFAULTS)[0]["action"] == "flagged"


def test_low_similarity_needs_a_real_title_and_latin_text_and_no_title_words_in_the_abstract():
    out = decide([
        row("Hazardous waste risk detection with machine learning sensors", "The weather in spring is mild and the garden blooms with many colourful flowers every year.", 0.40),   # suspect
        row("Sasha", "Smart home assistants function best when user commands are direct and well specified such as turn on the light.", 0.49),                                # name-only title: cannot judge
        row("Deep learning for graph anomaly detection benchmarks", "본 논문에서는 그래프 이상 탐지를 위한 딥러닝 벤치마크를 제안한다. 기존 방법은 한계가 있었다. 이를 극복하기 위해 새로운 데이터셋을 구축했다.", 0.45),  # not Latin
        row("Graph anomaly detection with contrastive learning", "Graph anomaly detection is hard; we use contrastive learning to find unusual nodes in large graphs of users.", 0.52),     # words overlap: ok
    ], DEFAULTS)
    assert out[0]["action"] == "suspect" and out[3]["action"] == "ok"
    assert out[1]["action"] in ("ok", "flagged") and out[2]["action"] in ("ok", "flagged")      # reported at most, never changed


def test_helpers():
    assert lexical_overlap("graph neural networks for chemistry", "chemistry with graph networks") >= 0.75
    assert title_in_abstract("Textbooks Are All You Need", "we use high quality textbook data") > 0
    assert is_latin("English text") and not is_latin("본 논문에서는 제안한다 한국어 초록입니다")
