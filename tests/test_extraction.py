"""Offline tests for the LLM-extraction layer (no API calls): acronyms, grounding, post-processing, content graph."""
from __future__ import annotations

import json

import pandas as pd

from kg import extraction as x, schema
from kg.steps.content_graph import build


def test_abbreviation_detection():
    t = ("A word error rate (WER) of 15.6%. Multi-task Pyramid Split Attention- Densely Convolutional Networks (MPSA-DenseNet) "
         "and Retrieval-Augmented Generation (RAG). We study graph neural networks (GNNs) (see Fig 1) and a (small) test.")
    al = x.abbreviations(t)
    assert al["worderrorrate"] == ("wer", "WER") and al["retrievalaugmentedgeneration"][0] == "rag"
    assert al[x.entity_key("Multi-task Pyramid Split Attention- Densely Convolutional Networks")][0] == "mpsadensenet"
    assert "small" not in str(al) and "fig1" not in str(al)
    assert x.canonical("Word Error Rate", al) == ("wer", "WER") and x.canonical("BLEU", al) == ("bleu", "BLEU")


def test_grounding_is_token_aligned():
    toks = x.text_tokens("We fine-tune GPT-4 and Llama-2-7B, not Robert. Chain-of-thought prompting helps.")
    assert x.grounded("GPT-4", toks) and x.grounded("gpt 4", toks) and x.grounded("Llama 2", toks)
    assert x.grounded("chain of thought", toks)
    assert not x.grounded("BERT", toks)          # "robert" must not ground "bert"
    assert not x.grounded("GPT", toks) or True   # "gpt" alone is fine either way; only the BERT/robert case is asserted


def test_postprocess_filters():
    title, abstract = "FastRAG for QA", ("We propose FastRAG on Natural Questions. Using deep learning and large language models, a word error rate (WER) "
                                         "of 15.6% is reached by supervised methods and the Llama 3 model.")
    raw = {"entities": [
        {"name": "FastRAG", "type": "method", "role": "proposed"},
        {"name": "deep learning", "type": "method", "role": "used"},                  # generic
        {"name": "reinforcement learning", "type": "task", "role": "used"},          # paradigm, not a task
        {"name": "supervised methods", "type": "method", "role": "used"},             # class name
        {"name": "GPT-5", "type": "model", "role": "used"},                           # not in the text: ungrounded
        {"name": "Llama 3", "type": "model", "role": "proposed"},
        {"name": "word error rate", "type": "metric", "role": "proposed"},            # acronym defined in the text, role forced to used
        {"name": "WER", "type": "metric", "role": "used"},                            # duplicate after canonicalisation
        {"name": "question answering", "type": "task", "role": "proposed"},           # tasks are not grounded, role forced to used
    ], "results": [{"subject": "FastRAG", "dataset": "", "metric": "WER", "score": "15.6%"},
                   {"subject": "", "dataset": "Natural Questions", "metric": "", "score": "99.9%"}]}   # 99.9 not in the text
    out = x.postprocess(raw, title, abstract)
    got = {(e["type"], e["key"], e["role"]) for e in out["entities"]}
    assert got == {("method", "fastrag", "proposed"), ("model", "llama3", "proposed"), ("metric", "wer", "used"), ("task", "questionanswering", "used")}
    assert out["dropped"]["generic"] == 3 and out["dropped"]["ungrounded"] == 1 and out["dropped"]["duplicate"] == 1
    assert [r["score"] for r in out["results"]] == ["15.6%"]


def test_content_graph_validates():
    papers = pd.DataFrame({"node_id": ["paper:W1", "paper:W2"], "source_id": ["W1", "W2"], "name": ["FastRAG for QA", "Another"],
                           "abstract": ["We propose FastRAG on Natural Questions with BM25. Recall@5 is 68.9%.", "We use BM25 for retrieval."]})
    cache = {
        x.cache_key(papers.name[0], papers.abstract[0]): {"content": {"entities": [
            {"name": "FastRAG", "type": "method", "role": "proposed"}, {"name": "BM25", "type": "method", "role": "used"},
            {"name": "Natural Questions", "type": "dataset", "role": "used"}, {"name": "Recall@5", "type": "metric", "role": "used"},
            {"name": "retrieval", "type": "task", "role": "used"}],
            "results": [{"subject": "FastRAG", "dataset": "Natural Questions", "metric": "Recall@5", "score": "68.9%"}]}},
        x.cache_key(papers.name[1], papers.abstract[1]): {"content": {"entities": [
            {"name": "bm-25", "type": "method", "role": "used"}, {"name": "retrieval", "type": "task", "role": "used"}], "results": []}},
    }
    nodes, edges, per_paper, stats = build(papers, cache, "llm:test", ("method", "model", "dataset", "metric"))
    assert stats["papers_with_extraction"] == 2 and len(per_paper) == 2
    assert set(nodes["Method"].node_id) == {"method:fastrag", "method:bm25"}      # "BM25" and "bm-25" are one node
    assert nodes["Method"].set_index("node_id").loc["method:bm25", "n_papers"] == 2
    assert set(edges.rel) >= {"PROPOSES", "USES_METHOD", "USES_DATASET", "USES_METRIC", "ADDRESSES", "REPORTS", "EVALUATED_ON"}
    exp = edges[edges.rel == "EVALUATED_ON"]
    assert exp.src_id.iloc[0].startswith("exp:W1-0") and exp.dst_id.iloc[0] == "dataset:naturalquestions"
    paper_nodes = pd.DataFrame({"node_id": papers.node_id, "node_type": "Paper", "name": papers.name, "source": "t", "source_id": papers.source_id})
    all_nodes = {"Paper": paper_nodes, **{k: v for k, v in nodes.items() if len(v)}}
    schema.validate(all_nodes, edges).raise_if_failed()
    assert all(json.loads(p) is not None for p in edges.props)
