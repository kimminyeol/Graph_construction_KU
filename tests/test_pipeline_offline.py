"""Offline end-to-end test: fake OpenAlex records -> select -> build -> export -> stats."""
from __future__ import annotations

import json

import networkx as nx
import pandas as pd
import pytest

from kg import schema
from kg.config import Config
from kg.io import append_jsonl
from kg.steps import build_base_graph, export_graph, graph_stats, select_corpus
from kg.steps.collect_openalex import plan_units
from kg.steps.select_corpus import merge_versions
from kg.steps.verify_metadata import arxiv_id, check_record_age, judge

OA = "https://openalex.org/"
ABSTRACT = {w: [i] for i, w in enumerate(f"word{i}" for i in range(50))}


def topic(tid, score, sf="1702", sf_name="Artificial Intelligence"):
    return {"id": OA + tid, "display_name": f"Topic {tid}", "score": score,
            "subfield": {"id": OA + "subfields/" + sf, "display_name": sf_name},
            "field": {"id": OA + "fields/17", "display_name": "Computer Science"},
            "domain": {"id": OA + "domains/3", "display_name": "Physical Sciences"}}


def work(wid, year, refs, authors, topics, cited=10, venue="S1"):
    return {
        "id": OA + wid, "title": f"Paper <i>{wid}</i>", "publication_year": year, "type": "article",
        "doi": f"https://doi.org/10.1/{wid}", "cited_by_count": cited, "referenced_works_count": len(refs),
        "referenced_works": [OA + r for r in refs], "abstract_inverted_index": ABSTRACT,
        "primary_location": {"source": {"id": OA + venue, "display_name": f"Venue {venue}", "type": "journal"}},
        "open_access": {"is_oa": True, "oa_url": "http://x"},
        "authorships": authors, "primary_topic": topics[0], "topics": topics,
    }


def author(aid=None, name="A", orcid=None, inst=("I1", "KAIST", "KR")):
    return {"author_position": "first", "is_corresponding": False,
            "author": {"id": OA + aid if aid else None, "display_name": name, "orcid": orcid},
            "institutions": [{"id": OA + inst[0], "display_name": inst[1], "country_code": inst[2],
                              "type": "education"}] if inst else []}


WORKS = [
    work("W1", 2022, refs=["W99"], topics=[topic("T1", 0.99), topic("T2", 0.3)],
         authors=[author("A1", "Alice"), author(None, "Bob Lee", orcid="https://orcid.org/0000-0001")]),
    work("W2", 2023, refs=["W1", "W98"], topics=[topic("T1", 0.9), topic("T3", 0.8, "1710", "Info Sys")],
         authors=[author("A1", "Alice"), author(None, "Carol Kim", inst=None)]),
    work("W3", 2024, refs=["W1", "W2", "W3"], topics=[topic("T3", 0.95, "1710", "Info Sys")],
         authors=[author("A2", "Dan", inst=("I2", "SNU", "KR"))], venue="S2"),
]


@pytest.fixture()
def cfg(tmp_path):
    data = {
        "run_name": "test",
        "paths": {"data_dir": str(tmp_path / "data"), "exports_dir": str(tmp_path / "exports")},
        "corpus": {"max_papers": 3, "years": [2022, 2024], "stratify_by_year": False,
                   "local_filters": {"min_abstract_words": 10},
                   "sources": [{"name": "t", "kind": "topic", "share": 1.0, "items": {"T1": "x", "T3": "y"}}]},
        "graph": {"citations": "internal", "max_topics_per_paper": 3, "min_topic_score": 0.5},
        "verify": {"enabled": False},  # registry lookups need the network; judge() is tested below
    }
    c = Config(data=data, path=tmp_path / "cfg.yaml")
    units = {u.item: u.key for u in plan_units(c)}
    raw = c.paths.openalex_raw / "works"
    # W2 appears under both units -> must be selected once, with both units recorded
    append_jsonl(raw / f"{units['T1']}.jsonl", [WORKS[0], WORKS[1]])
    append_jsonl(raw / f"{units['T3']}.jsonl", [WORKS[1], WORKS[2]])
    return c


def test_end_to_end(cfg):
    sel = select_corpus.run(cfg)
    assert sel["selected"] == 3
    corpus = pd.read_parquet(cfg.paths.processed / "corpus.parquet")
    assert len(corpus[corpus.paper_id == "W2"].corpus_units.iloc[0]) == 2

    build_base_graph.run(cfg)
    nodes, edges = export_graph.load_processed(cfg)
    papers = nodes["Paper"].set_index("node_id")

    # titles cleaned, abstract rebuilt
    assert papers.loc["paper:W1", "name"] == "Paper W1"
    assert papers.loc["paper:W1", "abstract"].startswith("word0 word1")

    # only internal citations, no self-citation
    cites = set(zip(edges[edges.rel == "CITES"].src_id, edges[edges.rel == "CITES"].dst_id))
    assert cites == {("paper:W2", "paper:W1"), ("paper:W3", "paper:W1"), ("paper:W3", "paper:W2")}
    assert papers.loc["paper:W3", "n_refs_internal"] == 2

    # author id fallbacks: OpenAlex id > ORCID > name
    a = nodes["Author"].set_index("node_id")
    assert a.loc["author:A1", "n_papers"] == 2
    assert a.loc["author:orcid-0000-0001", "id_source"] == "orcid"
    assert a.loc["author:name-carol-kim", "id_source"] == "name"

    # topic hierarchy + score threshold (T2 score 0.3 dropped)
    t = nodes["Topic"].set_index("node_id")
    assert "topic:T2" not in t.index
    assert set(t.level) == {"topic", "subfield", "field", "domain"}
    assert ("topic:SF1702", "topic:F17") in set(zip(edges.src_id, edges.dst_id))
    assert t.loc["topic:SF1702", "n_papers"] == 2 and t.loc["topic:F17", "n_papers"] == 3

    aff = edges[(edges.rel == "AFFILIATED_WITH") & (edges.src_id == "author:A1")].iloc[0]
    assert aff.weight == 2 and json.loads(aff.props) == {"first_year": 2022, "last_year": 2023}

    assert schema.validate(nodes, edges).ok

    export_graph.run(cfg)
    out = cfg.paths.exports
    g = nx.read_graphml(out / "graph.graphml")
    assert g.number_of_nodes() == sum(len(df) for df in nodes.values())
    assert g.number_of_edges() == len(edges)
    assert (out / "neo4j" / "nodes_Paper.csv").read_text().splitlines()[0].startswith("node_id:ID")
    assert "--relationships=edges_CITES.csv" in (out / "neo4j" / "import.sh").read_text()

    s = graph_stats.run(cfg)
    assert s["validation"]["ok"] and s["citations_internal"]["largest_wcc"] == 3


def _reg(title, authors, year=2022):
    return {"key": "x", "found": True, "title": title, "abstract": "real abstract", "year": year,
            "authors": authors}


def test_verify_judge_and_repair():
    vcfg, years = {}, (2022, 2026)
    w = work("W5", 2022, refs=[], topics=[topic("T1", 0.9)],
             authors=[author("A1", "Ouyang, Long"), author("A2", "Jeff Wu")])
    w["title"] = "Discernment and Social Learning as a Companion Training Layer"  # hijacked title

    ok = judge({**w, "title": "Training language models to follow instructions with human feedback"},
               _reg("Training Language Models to Follow Instructions with Human Feedback", ["Long Ouyang"]),
               "arxiv", vcfg, years)
    assert ok["status"] == "ok"

    rep = judge(w, _reg("Training language models to follow instructions with human feedback",
                        ["Long Ouyang", "Jeff Wu"]), "arxiv", vcfg, years)
    assert rep["status"] == "repaired" and rep["new_title"].startswith("Training language")

    assert judge(w, _reg("Some other paper", ["Rivo Kaugeranna"]), "datacite", vcfg, years)["status"] \
        == "drop_identity"

    old = judge({**w, "title": "Attention Is All You Need", "publication_year": 2025},
                _reg("Attention Is All You Need", ["Ouyang Long"], year=2017), "arxiv", vcfg, years)
    assert old["status"] == "drop_year" and old["new_year"] == 2017

    # MAG-era id (record existed by 2021) claiming 2025 -> overwritten year; 2023 is within the gap
    attn = judge({**w, "id": OA + "W2626778328", "title": "Attention Is All You Need", "publication_year": 2025},
                 _reg("Attention Is All You Need", ["Ouyang Long"], year=2025), "crossref", vcfg, years)
    assert attn["status"] == "ok" and check_record_age(attn, vcfg)["status"] == "drop_year"
    assert check_record_age({**attn, "oa_year": 2023}, vcfg)["status"] == "ok"
    assert check_record_age({**attn, "paper_id": "W4384918448"}, vcfg)["status"] == "ok"

    assert arxiv_id({"doi": "https://doi.org/10.48550/arxiv.2203.02155"}) == "2203.02155"
    assert arxiv_id({"primary_location": {"landing_page_url": "http://arxiv.org/abs/2303.08774v3"}}) == "2303.08774"

    nodes, _ = build_base_graph.build_tables([w], None, {}, {"W5": rep})
    p = nodes["Paper"].iloc[0]
    assert p["name"].startswith("Training language") and p["abstract"] == "real abstract"
    assert p["metadata_status"] == "repaired"


def test_merge_versions_and_citation_redirect():
    title = "A Survey on Hallucination in Large Language Models"
    meta = {"Wpub": {"title": title, "year": 2025, "has_refs": True, "is_article": True, "cited": 10},
            "Wpre": {"title": title + ".", "year": 2023, "has_refs": False, "is_article": False, "cited": 900},
            "Wold": {"title": title, "year": 2015, "has_refs": True, "is_article": True, "cited": 5},
            "Wshort": {"title": "Large Language Models", "year": 2024, "has_refs": True, "is_article": True,
                       "cited": 1}}
    canon = merge_versions(meta, {})
    assert canon["Wpre"] == "Wpub"            # reference list + article wins over raw citation count
    assert canon["Wold"] == "Wold"            # year gap too large: different paper
    assert canon["Wshort"] == "Wshort"        # generic short titles are never merged

    pub = work("Wpub", 2025, refs=[], topics=[topic("T1", 0.9)], authors=[author("A1")])
    citer = work("Wc", 2025, refs=["Wpre"], topics=[topic("T1", 0.9)], authors=[author("A2")])
    nodes, edges = build_base_graph.build_tables([pub, citer], None, {}, alt_ids={"Wpub": ["Wpre"]})
    assert set(zip(edges[edges.rel == "CITES"].src_id, edges[edges.rel == "CITES"].dst_id)) == \
        {("paper:Wc", "paper:Wpub")}
    assert list(nodes["Paper"].set_index("node_id").loc["paper:Wpub", "alt_ids"]) == ["Wpre"]


def test_validation_catches_bad_edges():
    nodes = {"Paper": pd.DataFrame([{"node_id": "paper:W1", "node_type": "Paper", "name": "x",
                                     "source": "t", "source_id": "W1"}])}
    bad = pd.DataFrame([
        {"src_id": "paper:W1", "rel": "AUTHORED", "dst_id": "paper:W1", "src_type": "Paper",
         "dst_type": "Paper", "weight": 1.0, "source": "t", "props": "{}"},
        {"src_id": "paper:W1", "rel": "CITES", "dst_id": "paper:W404", "src_type": "Paper",
         "dst_type": "Paper", "weight": 1.0, "source": "t", "props": "{}"},
    ])
    errors = " ".join(schema.validate(nodes, bad).errors)
    assert "AUTHORED: Paper->Paper not allowed" in errors
    assert "dangling" in errors


def test_synonym_edges_must_share_type():
    e = schema.EDGE_TYPES["SYNONYM_OF"]
    assert e.same_type and "Dataset" in e.src
    nodes = {t: pd.DataFrame([{"node_id": schema.make_id(t, "x"), "node_type": t, "name": "x",
                               "source": "t", "source_id": "x"}]) for t in ("Dataset", "Model")}
    edges = pd.DataFrame([{"src_id": "dataset:x", "rel": "SYNONYM_OF", "dst_id": "model:x",
                           "src_type": "Dataset", "dst_type": "Model", "weight": 1.0, "source": "t",
                           "props": "{}"}])
    assert not schema.validate(nodes, edges).ok
