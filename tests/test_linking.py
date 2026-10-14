"""Offline tests for entity linking (no API, no GPU): normalisation key, acronym tier, judge tier, cluster cap."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from kg import linking as L


def test_link_key_merges_variants_and_keeps_distinct_things_apart():
    k = L.link_key
    same = [("GNNs", "GNN"), ("summarisation of text", "text summarization"), ("GPT-4.0", "GPT-4"), ("token-level F1", "token-F1"),
            ("prediction of molecular properties", "molecular property prediction"), ("answering questions", "question answering"),
            ("LLM-based agents", "LLM agents"), ("neighbour", "neighbor"), ("ATT&CK™", "ATT&CK")]
    for a, b in same:
        assert k(a) == k(b), (a, b)
    assert k("test set", "dataset") == k("test dataset", "dataset") == k("test corpus", "dataset")
    different = [("ANNS", "ANN"), ("image-to-text", "text-to-image"), ("GPT-4", "GPT-4o"), ("Llama 2", "Llama 3"), ("optimizer", "optimization"),
                 ("generation-augmented retrieval", "retrieval-augmented generation"), ("policy-based", "policy-level"),
                 ("human decision-making", "human-AI decision-making"), ("YOLOv6", "YOLOv6-N")]
    for a, b in different:
        assert k(a) != k(b), (a, b)


def _ctx(names: dict[str, tuple[str, int]], acronyms=None) -> L.Ctx:
    nodes = pd.DataFrame([{"node_id": f"method:{n}", "name": nm, "n_papers": c, "typ": "method"} for n, (nm, c) in names.items()])
    papers_of = {f"method:{n}": {f"paper:{n}{i}" for i in range(c)} for n, (_, c) in names.items()}
    ctx = L.Ctx(nodes, papers_of)
    ctx.acronyms = acronyms or {}
    return ctx


def test_acronym_tier_needs_support_and_a_dominant_expansion():
    ctx = _ctx({"rag": ("RAG", 9), "rag2": ("retrieval-augmented generation", 3), "rs": ("RS", 4), "rs1": ("remote sensing", 2), "rs2": ("recommender systems", 2)},
               {"rag": Counter({"retrievalaugmentgener": 12}), "rs": Counter({"remotesens": 5, "recommendersystem": 5})})
    r = L.compute_links(ctx, {"tiers": ["t1", "t2"]})
    cl = r["clusters"]
    assert cl["method:rag"]["canonical"] == cl["method:rag2"]["canonical"] == "method:rag"       # the better-supported name is canonical
    assert len({cl[n]["canonical"] for n in ("method:rs", "method:rs1", "method:rs2")}) == 3     # ambiguous acronym: nothing merged
    assert cl["method:rag2"]["tier"] == "t2"


def test_judge_tier_and_cluster_cap():
    names = {f"x{i}": (f"widget method {chr(97 + i)}", 1) for i in range(5)}
    ctx = _ctx(names)
    vecs = lambda texts: np.tile(np.ones(4, dtype=np.float32) / 2, (len(texts), 1))              # every pair is a candidate
    calls = []
    def judge(pairs):
        calls.append(len(pairs))
        return [True] * len(pairs)
    r = L.compute_links(ctx, {"tiers": ["t3"], "embed_min_cos": 0.5, "max_cluster": 3, "embed_min_len": 3}, vecs, judge)
    sizes = Counter(c["canonical"] for c in r["clusters"].values())
    assert calls and max(sizes.values()) <= 3                                                      # the cap stops a chain from swallowing everything
    # without a judge, only names above the strict threshold are merged
    r2 = L.compute_links(ctx, {"tiers": ["t3"], "embed_min_cos": 0.5, "embed_min_cos_nojudge": 1.1, "embed_min_len": 3}, vecs, None)
    assert r2["stats"]["t3_unions"] == 0
