"""Offline tests for community summaries: evidence building (characteristic entities, neighbours, shown titles)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kg import summarize as S


def _tables():
    papers = pd.DataFrame({"node_id": [f"paper:{i}" for i in range(12)], "name": [f"title {i}" for i in range(12)], "year": 2024})
    comm = pd.DataFrame({"node_id": ["community:L0-0", "community:L0-1"], "name": ["a", "b"], "level": 0, "n_papers": 6, "median_year": 2024.0,
                         "keywords": [["rag", "retrieval"], ["gnn", "graph"]], "top_topics": [["Topic X (6)"], ["Topic Y (6)"]]})
    edges = pd.DataFrame([{"src_id": f"paper:{i}", "rel": "BELONGS_TO", "dst_id": f"community:L0-{i // 6}", "src_type": "Paper", "dst_type": "Community",
                           "weight": 1.0, "source": "leiden", "props": json.dumps({"level": 0})} for i in range(12)])
    vecs = {f"paper:{i}": (np.array([1.0, 0.1 * (i % 6)]) if i < 6 else np.array([0.1 * (i % 6), 1.0])) / 1.0 for i in range(12)}
    vecs = {k: v / np.linalg.norm(v) for k, v in vecs.items()}
    # entity "rag" is used by every paper of community 0 (characteristic); "accuracy" by all papers (a corpus staple)
    ext = pd.DataFrame([{"src_id": f"paper:{i}", "dst_id": "method:rag", "dst_type": "Method"} for i in range(6)]
                       + [{"src_id": f"paper:{i}", "dst_id": "metric:accuracy", "dst_type": "Metric"} for i in range(12)])
    links = pd.DataFrame([{"src_id": "paper:0", "dst_id": "paper:6", "rel": "CITES", "weight": 1.0}, {"src_id": "paper:1", "dst_id": "paper:7", "rel": "SIMILAR_TO", "weight": 0.9}])
    return {"comm": comm, "edges": edges, "papers": papers, "vecs": vecs, "ext": ext, "names": {"method:rag": "RAG", "metric:accuracy": "accuracy"}, "canon": {}, "links": links}


def test_build_inputs_picks_characteristic_entities_and_neighbours():
    inp = S.build_inputs(_tables(), n_central=2, n_random=1)
    p = inp["L0-0"]["payload"]
    assert p["characteristic_entities"]["methods"] == ["RAG (6)"]                    # used by 6 of 6 papers, absent elsewhere
    assert "metrics" not in p["characteristic_entities"]                              # a corpus-wide staple has lift 1: not characteristic
    assert len(inp["L0-0"]["shown"]) == 3 and set(inp["L0-0"]["shown"]) <= set(inp["L0-0"]["members"])
    assert len(p["example_titles_central"]) == 2 and len(p["example_titles_random"]) == 1
    assert p["closest_communities"][0].endswith("(100% of links)") and "gnn" in p["closest_communities"][0]
    again = S.build_inputs(_tables(), n_central=2, n_random=1)
    assert again["L0-0"]["shown"] == inp["L0-0"]["shown"]                             # deterministic across processes
