"""Offline tests: author merge, arXiv-reference matching, version merging."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kg.steps.arxiv_refs import CorpusIndex, match_entry
from kg.steps.build_base_graph import author_canonical_map
from kg.steps.select_corpus import merge_versions_v2

OA = "https://openalex.org/"


def authorship(aid, name, orcid=None):
    return {"author": {"id": OA + aid if aid else None, "display_name": name, "orcid": f"https://orcid.org/{orcid}" if orcid else None},
            "institutions": []}


def test_author_merge_needs_compatible_names():
    works = [
        {"id": OA + "W1", "authorships": [authorship("A1", "Aixin Sun", "0000-0003-0764-4258"), authorship("A2", "Chenliang Li", None)]},
        {"id": OA + "W2", "authorships": [authorship("A3", "Aixin Sun", "0000-0003-0764-4258")]},
        # ORCID attached to the wrong co-author (no OpenAlex id): must not be merged with the OpenAlex node of another person
        {"id": OA + "W3", "authorships": [authorship(None, "Chenliang Li", "0000-0003-0764-4258")]},
        {"id": OA + "W4", "authorships": [authorship("A4", "Yong Zhang", "0000-0003-0066-3448"), authorship(None, "Baoyuan Wu", "0000-0003-0066-3448")]},
    ]
    canon = author_canonical_map(works)
    assert canon.get("author:A3") == "author:A1" or canon.get("author:A1") == "author:A3"   # same person, two OpenAlex ids
    assert "author:orcid-0000-0003-0764-4258" not in canon                                    # Chenliang Li != Aixin Sun
    assert "author:orcid-0000-0003-0066-3448" not in canon and "author:A4" not in canon      # swapped ORCID: nothing merged


def _index():
    papers = pd.DataFrame({"source_id": ["W1", "W2", "W3"],
                           "name": ["Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "Attention Is All You Need",
                                    "Large Language Models: A Survey"], "year": [2023, 2017, 2024]})
    def w(arx, first):
        return {"doi": f"https://doi.org/10.48550/arxiv.{arx}", "primary_location": {"landing_page_url": f"http://arxiv.org/abs/{arx}"},
                "authorships": [authorship(None, first)]}
    works = {"W1": w("2305.10601", "Shunyu Yao"), "W2": w("1706.03762", "Ashish Vaswani"), "W3": w("2402.06196", "Shervin Minaee")}
    return CorpusIndex(papers, works, {})


def test_reference_matching_rules():
    idx = _index()
    e = lambda text, **k: {"text": text, "arxiv_ids": k.get("arxiv", []), "dois": []}
    # title found in the entry
    m = match_entry(e("Yao et al. (2023) Shunyu Yao, Dian Yu. 2023. Tree of thoughts: Deliberate problem solving with large language models. NeurIPS."), idx)
    assert [x["pid"] for x in m] == ["W1"]
    # id found even when the title is mangled
    assert [x["pid"] for x in match_entry(e("Vaswani et al. Attn. 2017", arxiv=["1706.03762"]), idx)] == ["W2"]
    # a generic 5-word title needs the first author's surname: "A survey of large language models" by Zhao must not hit Minaee's paper
    assert match_entry(e("Zhao et al. 2023. Large language models: a survey. arXiv"), idx) == []
    assert [x["pid"] for x in match_entry(e("Minaee et al. 2024. Large language models: a survey. arXiv"), idx)] == ["W3"]
    # self-citation is dropped
    assert match_entry(e("Yao. 2023. Tree of thoughts: Deliberate problem solving with large language models"), idx, self_pid="W1") == []


def test_version_merging_needs_authors_and_type_pattern():
    base = {"has_refs": True, "is_article": False, "is_preprint": True, "cited": 1}
    meta = {
        "A": {**base, "title": "Explainable AI in medical imaging an overview for clinicians", "year": 2022, "authors": ["Ann One", "Bob Two"]},
        "B": {**base, "title": "Explainable AI in medical imaging an overview for clinicians", "year": 2023, "authors": ["Carl Three", "Dee Four"]},
        "C": {**base, "title": "GatorTron a large clinical language model", "year": 2022, "authors": ["Xi Yang", "Jiang Bian"], "is_preprint": True},
        "D": {**base, "title": "A large language model for electronic health records", "year": 2022, "authors": ["Xi Yang", "Jiang Bian"], "is_preprint": False, "is_article": True},
        "E": {**base, "title": "Qwen2 technical report", "year": 2024, "authors": ["An Yang", "Baosong Yang", "Chang Zhou", "Dayi Liu"]},
        "F": {**base, "title": "Qwen2.5 technical report", "year": 2024, "authors": ["An Yang", "Baosong Yang", "Chang Zhou", "Zhi Li"]},
    }
    v = {k: np.zeros(4, dtype=np.float32) for k in meta}
    for k, vec in {"A": [1, 0, 0, 0], "B": [1, 0, 0, 0], "C": [0, 1, 0, 0], "D": [0, 1, 0, 0], "E": [0, 0, 1, 0], "F": [0, 0, 1, 0]}.items():
        v[k] = np.array(vec, dtype=np.float32)
    canon, stats = merge_versions_v2(meta, {"title_author_overlap": 0.5, "embedding_cos": 0.95, "author_overlap": 0.6, "same_type_author_overlap": 0.95}, v)
    assert canon["A"] != canon["B"]          # same title, different authors: not the same paper (v1 merged them)
    assert canon["C"] == canon["D"] == "D"   # preprint + article with the same authors, different title; the article is canonical
    assert canon["E"] != canon["F"]          # preprint / preprint successors with different authors
    assert stats["r1_blocked_by_authors"] == 1 and stats["r2_embedding"] == 1


def test_same_abstract_rule_merges_renamed_versions_with_shared_authors():
    abs_ = "We present a unified codec language model for speech recognition, synthesis and translation, trained on many tasks " * 2
    base = {"has_refs": True, "is_article": False, "is_preprint": True, "cited": 1, "year": 2023, "abstract": abs_}
    meta = {"A": {**base, "title": "VioLA conditional language models for speech", "authors": ["Tianrui Wang", "Long Zhou"]},
            "B": {**base, "title": "VioLA unified codec language models for speech", "authors": ["Tianrui Wang", "Long Zhou"], "is_preprint": False, "is_article": True},
            "C": {**base, "title": "Another paper with a copied abstract", "authors": ["Someone Else", "Other Person"]}}
    canon, stats = merge_versions_v2(meta, {"title_author_overlap": 0.5, "same_abstract": True})
    assert canon["A"] == canon["B"] == "B" and canon["C"] == "C"
    assert stats["r3_same_abstract"] == 1 and stats["r3_blocked_by_authors"] == 2
