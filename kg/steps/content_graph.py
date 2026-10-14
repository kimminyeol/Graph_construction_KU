"""Step `content`: cached LLM extractions -> Method / Model / Dataset / Task / Metric / Experiment nodes + edges.

Reads data/raw/llm_extract/<model>__<prompt>.jsonl (written by `extract`), applies `extraction.postprocess`
(grounding, generic-term filter, dedupe) and merges mentions that normalise to the same name (case / hyphen / space).
Merging *different* names for the same thing (BERT-base vs BERT, abbreviation vs full name) is entity linking (Phase 4).

Output (all next to the base tables; `export` / `stats` pick them up through `load_processed`):
  processed/<run>/{methods,models,datasets,tasks,metrics,experiments}_ext.parquet
  processed/<run>/edges_extract.parquet        PROPOSES / USES_* / ADDRESSES / REPORTS / EVALUATED_ON / USES_METRIC
  processed/<run>/extractions.parquet          one post-processed record per paper (audit / evaluation)
  processed/<run>/content_report.json

The `_ext` suffix keeps these tables apart from the Hugging Face ones (`models.parquet`, ...): the two graph variants
(with / without HF) differ only in which files are present.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict

import pandas as pd

from .. import extraction, schema
from ..config import Config
from ..io import iter_jsonl, read_parquet, write_json, write_parquet
from .extract_entities import cache_file

log = logging.getLogger(__name__)

NODE_OF = {"task": "Task", "method": "Method", "model": "Model", "dataset": "Dataset", "metric": "Metric"}
REL_OF = {("task", "used"): "ADDRESSES", ("method", "used"): "USES_METHOD", ("model", "used"): "USES_MODEL",
          ("dataset", "used"): "USES_DATASET", ("metric", "used"): "USES_METRIC"}


def rel_for(typ: str, role: str) -> str:
    return "PROPOSES" if role == "proposed" else REL_OF[(typ, "used")]


def _frame(node_type: str, rows: list[dict]) -> pd.DataFrame:
    nt = schema.NODE_TYPES[node_type]
    cols = [*schema.NODE_BASE_COLUMNS, *nt.properties]
    return pd.DataFrame(rows, columns=cols)


def build(papers: pd.DataFrame, cache: dict[str, dict], source: str, ground: tuple[str, ...]) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    ents: dict[tuple[str, str], dict] = {}
    edges: list[dict] = []
    exps: list[dict] = []
    per_paper: list[dict] = []
    dropped, missing = Counter(), 0

    def entity(typ: str, key: str, name: str, pid: str):
        e = ents.setdefault((typ, key), {"surfaces": Counter(), "papers": set()})
        e["surfaces"][name] += 1
        e["papers"].add(pid)
        return schema.make_id(NODE_OF[typ], key)

    def edge(src: str, rel: str, dst: str, dst_type: str, **props):
        src_type = schema.type_of(src)
        edges.append({"src_id": src, "rel": rel, "dst_id": dst, "src_type": src_type, "dst_type": dst_type,
                      "weight": 1.0, "source": source, "props": json.dumps(props, ensure_ascii=False)})

    for r in papers.itertuples():
        rec = cache.get(extraction.cache_key(r.name, r.abstract))
        if rec is None:
            missing += 1
            continue
        out = extraction.postprocess(rec["content"], r.name, r.abstract, ground)
        dropped.update(out["dropped"])
        per_paper.append({"paper_id": r.node_id, "entities": json.dumps(out["entities"], ensure_ascii=False),
                          "results": json.dumps(out["results"], ensure_ascii=False), "dropped": json.dumps(out["dropped"])})
        mine: dict[tuple[str, str], tuple[str, str]] = {}      # (type, key) -> (node_id, role) within this paper
        for e in out["entities"]:
            nid = entity(e["type"], e["key"], e["name"], r.node_id)
            mine[(e["type"], e["key"])] = (nid, e["role"])
            edge(r.node_id, rel_for(e["type"], e["role"]), nid, NODE_OF[e["type"]], surface=e["name"])
        for i, x in enumerate(out["results"]):
            eid = schema.make_id("Experiment", f"{r.source_id}-{i}")
            exps.append({"node_id": eid, "node_type": "Experiment", "name": f"{x['subject'] or 'paper'} | {x['dataset'] or '-'} | {x['metric'] or '-'} = {x['score']}",
                         "source": source, "source_id": f"{r.source_id}-{i}", "paper_id": r.node_id, "score": x["score"], "setting": None})
            edge(r.node_id, "REPORTS", eid, "Experiment")
            for typ, field, rel in (("dataset", "dataset", "EVALUATED_ON"), ("metric", "metric", "USES_METRIC")):
                if x[field]:
                    key = extraction.entity_key(x[field])
                    if (typ, key) not in mine:       # named only in the result: still a (used) entity of the paper
                        nid = entity(typ, key, x[field], r.node_id)
                        mine[(typ, key)] = (nid, "used")
                        edge(r.node_id, rel_for(typ, "used"), nid, NODE_OF[typ], surface=x[field])
                    edge(eid, rel, mine[(typ, key)][0], NODE_OF[typ])
            if x["subject"]:
                key = extraction.entity_key(x["subject"])
                for typ, rel in (("method", "USES_METHOD"), ("model", "USES_MODEL")):
                    if (typ, key) in mine:
                        edge(eid, rel, mine[(typ, key)][0], NODE_OF[typ])
                        break

    nodes = {}
    for typ, node_type in NODE_OF.items():
        rows = []
        for (t, key), e in sorted(ents.items()):
            if t != typ:
                continue
            ranked = [s for s, _ in e["surfaces"].most_common()]
            rows.append({"node_id": schema.make_id(node_type, key), "node_type": node_type, "name": ranked[0], "source": source,
                         "source_id": key, "aliases": ranked[1:11], "n_papers": len(e["papers"])})
        nodes[node_type] = _frame(node_type, rows)
    nodes["Experiment"] = _frame("Experiment", exps)
    edge_df = pd.DataFrame(edges, columns=list(schema.EDGE_COLUMNS))
    stats = {"papers_with_extraction": len(per_paper), "papers_without_extraction": missing, "dropped": dict(dropped)}
    return nodes, edge_df, pd.DataFrame(per_paper), stats


def run(cfg: Config) -> dict:
    ecfg = cfg.get("extract", {})
    path = cache_file(cfg, ecfg) if ecfg.get("model") else None
    if path is None or not path.exists():
        log.info("content: no extraction cache (%s); skipped", path)
        return {"skipped": True}
    cache = {r["key"]: r for r in iter_jsonl(path)}
    papers = read_parquet(cfg.paths.processed / "papers.parquet")[["node_id", "source_id", "name", "abstract"]]
    papers = papers[papers["name"].notna() & papers["abstract"].notna()]
    ground = tuple(ecfg.get("ground", ("method", "model", "dataset", "metric")))
    nodes, edges, per_paper, stats = build(papers, cache, f"llm:{ecfg['model']}", ground)

    out = cfg.paths.processed
    for node_type, df in nodes.items():
        write_parquet(df, out / f"{schema.NODE_TYPES[node_type].table}_ext.parquet")
    write_parquet(edges, out / "edges_extract.parquet")
    write_parquet(per_paper, out / "extractions.parquet")

    n = max(1, stats["papers_with_extraction"])
    per_type = {}
    for node_type, df in nodes.items():
        if node_type == "Experiment":
            per_type[node_type] = {"nodes": len(df)}
            continue
        per_type[node_type] = {"nodes": len(df), "single_paper_share": round(float((df["n_papers"] == 1).mean()), 3) if len(df) else None,
                               "top": [f"{r['name']} ({r['n_papers']})" for _, r in df.nlargest(12, "n_papers").iterrows()]}
    by_rel = edges["rel"].value_counts().to_dict()
    per_paper_cov = {}
    if len(edges):
        pe = edges[edges.src_type == "Paper"]
        for typ, node_type in NODE_OF.items():
            per_paper_cov[typ] = round(pe[pe.dst_type == node_type].src_id.nunique() / n, 3)
    report = {"model": ecfg["model"], **stats, "nodes": per_type, "edges": by_rel, "papers_with_at_least_one": per_paper_cov,
              "mean_entities_per_paper": round(len(edges[edges.src_type == "Paper"]) / n, 2) if len(edges) else 0}
    write_json(out / "content_report.json", report)
    log.info("content: %s", {k: v for k, v in report.items() if k != "nodes"})
    return {"edges": by_rel, "nodes": {t: v["nodes"] for t, v in per_type.items()}}
