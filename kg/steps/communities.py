"""Step `communities`: Leiden communities on a Paper-Paper projection of the KG.

Projection weights (configurable under `communities.weights`):
  CITES, SIMILAR_TO (x its cosine similarity), and papers sharing a Model / Dataset / Method / Task
  (via PROPOSES/USES_*/ADDRESSES/EVALUATED_ON) or an OpenAlex Topic. Shared-node links skip "hub"
  nodes that touch more than `max_hub_papers` papers (e.g. the topic "Topic Modeling"), which would
  otherwise link thousands of unrelated papers. With `hub_normalize` (default) each shared node
  contributes w/(n-1) per pair, i.e. at most `w` in total per paper.

Output: processed/<run>/communities.parquet (Community nodes with keywords),
        edges_comm.parquet (Paper -BELONGS_TO-> Community), communities_report.json
Quality report: modularity, size distribution, share of edge weight kept inside communities by
source, and agreement (NMI / purity) with OpenAlex subfields, which the algorithm never sees.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from itertools import combinations

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd

from .. import schema
from ..config import Config
from ..io import read_parquet, write_json, write_parquet
from .build_base_graph import _nodes_frame
from .export_graph import load_processed

log = logging.getLogger(__name__)
SOURCE = "leiden"

DEFAULT_WEIGHTS = {"CITES": 1.0, "SIMILAR_TO": 1.0, "shared_model": 1.0, "shared_dataset": 1.0,
                   "shared_method": 1.0, "shared_task": 1.0, "shared_topic": 0.5}
# (weight key, rel, hub node type): papers pointing at the same hub node are linked
SHARED = (("shared_model", ("PROPOSES", "USES_MODEL"), "Model"),
          ("shared_dataset", ("PROPOSES", "USES_DATASET"), "Dataset"),
          ("shared_method", ("PROPOSES", "USES_METHOD"), "Method"),
          ("shared_task", ("ADDRESSES",), "Task"),
          ("shared_topic", ("HAS_TOPIC",), "Topic"))
STOP = set("""a an and are as at be by for from has have in into is it its of on or that the their this to
with we our using use based via can how what which not new paper approach method methods model models study
analysis results result also these such than more between both other can may been were was while their
data system systems learning large language""".split())


def project(nodes: dict[str, pd.DataFrame], edges: pd.DataFrame, weights: dict,
            max_hub: int, hub_normalize: bool = True) -> tuple[dict[tuple[str, str], float], dict[str, dict[tuple[str, str], float]]]:
    """Paper-Paper weights, plus the same weights split by contribution source."""
    paper_ids = set(nodes["Paper"]["node_id"])
    total: dict[tuple[str, str], float] = defaultdict(float)
    by_src: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))

    def add(a: str, b: str, w: float, src: str) -> None:
        if a != b and w:
            key = (a, b) if a < b else (b, a)
            total[key] += w
            by_src[src][key] += w

    for rel in ("CITES", "SIMILAR_TO"):
        w = weights.get(rel, 0)
        for r in edges[edges.rel == rel].itertuples():
            add(r.src_id, r.dst_id, w * (r.weight if rel == "SIMILAR_TO" else 1.0), rel)

    # linked entities: an alias counts as its canonical entity (SYNONYM_OF alias -> canonical), so "RAG" and
    # "retrieval-augmented generation" are one shared hub
    syn = edges[edges.rel == "SYNONYM_OF"]
    canon = dict(zip(syn.src_id, syn.dst_id))
    for key, rels, hub_type in SHARED:
        w = weights.get(key, 0)
        if not w or hub_type not in nodes:
            continue
        sub = edges[edges.rel.isin(rels) & edges.src_id.isin(paper_ids) & (edges.dst_type == hub_type)]
        # Task is reached through the Models/Datasets a paper proposes: Paper -> Model -> Task
        # (and, for the Hugging Face layer, through the Models/Datasets a paper proposes: Paper -> Model -> Task)
        if hub_type == "Task":
            via = edges[edges.rel.isin(("PROPOSES",)) & edges.src_id.isin(paper_ids)][["src_id", "dst_id"]]
            tsk = edges[(edges.rel == "ADDRESSES") & (edges.dst_type == "Task") & ~edges.src_id.isin(paper_ids)][["src_id", "dst_id"]]
            two = via.merge(tsk, left_on="dst_id", right_on="src_id", suffixes=("", "_t"))[["src_id", "dst_id_t"]]
            two.columns = ["src_id", "dst_id"]
            sub = pd.concat([sub[["src_id", "dst_id"]], two], ignore_index=True)
        sub = sub.assign(dst_id=sub.dst_id.map(lambda x: canon.get(x, x)))
        for hub, g in sub.groupby("dst_id"):
            members = sorted(set(g.src_id))
            if 2 <= len(members) <= max_hub:
                # normalised: each shared node gives a paper at most `w` in total, however many papers
                # it touches, so a 200-paper topic cannot outweigh content similarity or citations
                pw = w / (len(members) - 1) if hub_normalize else w
                for a, b in combinations(members, 2):
                    add(a, b, pw, key)
    return total, by_src


def run_leiden(paper_ids: list[str], weights: dict[tuple[str, str], float], resolution: float,
               seed: int, n_iter: int = -1) -> tuple[list[int], float]:
    idx = {p: i for i, p in enumerate(paper_ids)}
    g = ig.Graph(n=len(paper_ids), edges=[(idx[a], idx[b]) for a, b in weights], directed=False)
    g.es["weight"] = list(weights.values())
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, n_iterations=n_iter, seed=seed)
    return part.membership, float(part.modularity)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower()) if t not in STOP]


def keywords(texts_by_comm: dict[int, list[str]], top_n: int = 8, min_docs: int = 3) -> dict[int, list[str]]:
    """Terms (uni+bigrams) over-represented in a community's titles vs. all titles (log-odds-ish)."""
    def terms(t: str) -> set[str]:
        w = _tokens(t)
        return set(w) | {f"{a} {b}" for a, b in zip(w, w[1:])}
    doc_terms = {c: [terms(t) for t in ts] for c, ts in texts_by_comm.items()}
    df_all = Counter(t for docs in doc_terms.values() for d in docs for t in d)
    n_all = sum(len(d) for d in doc_terms.values())
    out = {}
    for c, docs in doc_terms.items():
        df_c = Counter(t for d in docs for t in d)
        scored = []
        for t, n in df_c.items():
            if n < min(min_docs, len(docs)):
                continue
            p_in, p_all = n / len(docs), df_all[t] / n_all
            scored.append((p_in * math.log(p_in / p_all) * (1 + 0.3 * (" " in t)), t))
        out[c] = [t for _, t in sorted(scored, reverse=True)[:top_n]]
    return out


def nmi_purity(a: list, b: list) -> tuple[float, float]:
    """NMI(a, b) and purity of partition a w.r.t. labels b."""
    df = pd.DataFrame({"a": a, "b": b})
    n = len(df)
    ct = pd.crosstab(df.a, df.b).to_numpy(dtype=float)
    pa, pb = ct.sum(1) / n, ct.sum(0) / n
    pab = ct / n
    nz = pab > 0
    mi = float((pab[nz] * np.log(pab[nz] / np.outer(pa, pb)[nz])).sum())
    ha, hb = -float((pa * np.log(pa)).sum()), -float((pb * np.log(pb)).sum())
    return (mi / ((ha + hb) / 2) if ha + hb else 1.0), float(ct.max(1).sum() / n)


def _level(level: int, resolution: float, paper_ids, papers, total, by_src, nodes, edges, ccfg) -> tuple:
    min_size = ccfg.get("min_size", 5)
    membership, modularity = run_leiden(paper_ids, total, resolution, ccfg.get("seed", 42))
    sizes = Counter(membership)
    order = [c for c, n in sizes.most_common() if n >= min_size]   # communities below min_size stay unassigned
    label = {c: i for i, c in enumerate(order)}
    comm_of = {p: label.get(m, -1) for p, m in zip(paper_ids, membership)}
    log.info("communities L%d: resolution %.2f -> %d communities (+%d papers in tiny ones), modularity %.3f",
             level, resolution, len(order), sum(n for c, n in sizes.items() if n < min_size), modularity)

    texts: dict[int, list[str]] = defaultdict(list)
    for p, t in zip(paper_ids, papers["name"]):
        if comm_of[p] >= 0:
            texts[comm_of[p]].append(t)
    kws = keywords(texts, top_n=8)
    topic_name = nodes["Topic"].set_index("node_id")["name"] if "Topic" in nodes else pd.Series(dtype=str)
    prim = papers["primary_topic_id"].map(topic_name)
    members_of = defaultdict(list)
    for i, p in enumerate(paper_ids):
        if comm_of[p] >= 0:
            members_of[comm_of[p]].append(i)
    rows = []
    for c in range(len(order)):
        m = members_of[c]
        sid = f"L{level}-{c}"
        rows.append({"node_id": schema.make_id("Community", sid), "node_type": "Community",
                     "name": ", ".join(kws[c][:3]) if kws[c] else f"community {sid}", "source": SOURCE,
                     "source_id": sid, "level": level, "summary": None, "keywords": kws[c], "n_papers": len(m),
                     "top_topics": [f"{t} ({n})" for t, n in prim.iloc[m].value_counts().head(3).items()],
                     "median_year": float(papers["year"].iloc[m].median())})
    belongs = [{"src_id": p, "rel": "BELONGS_TO", "dst_id": schema.make_id("Community", f"L{level}-{c}"),
                "src_type": "Paper", "dst_type": "Community", "weight": 1.0, "source": SOURCE,
                "props": json.dumps({"resolution": resolution, "level": level})}
               for p, c in comm_of.items() if c >= 0]

    inside = {}
    for src, ws in by_src.items():
        tot = sum(ws.values())
        inside[src] = round(sum(w for (a, b), w in ws.items() if comm_of[a] == comm_of[b] and comm_of[a] >= 0)
                            / tot, 3) if tot else None
    t2sf = {}
    if "Topic" in nodes:
        sub = edges[edges.rel == "SUBTOPIC_OF"]
        sf_ids = set(nodes["Topic"].loc[nodes["Topic"].level == "subfield", "node_id"])
        t2sf = {a: b for a, b in zip(sub.src_id, sub.dst_id) if b in sf_ids}
    sf = papers["primary_topic_id"].map(t2sf).fillna("none").tolist()
    tp = papers["primary_topic_id"].fillna("none").tolist()
    mask = [i for i, p in enumerate(paper_ids) if comm_of[p] >= 0]
    nmi_sf, pur_sf = nmi_purity([comm_of[paper_ids[i]] for i in mask], [sf[i] for i in mask])
    nmi_tp, pur_tp = nmi_purity([comm_of[paper_ids[i]] for i in mask], [tp[i] for i in mask])
    sz = sorted((sizes[c] for c in order), reverse=True)
    report = {"level": level, "resolution": resolution, "communities": len(order), "modularity": round(modularity, 4),
              "papers_assigned": len(mask), "papers_unassigned": len(paper_ids) - len(mask),
              "size_top10": sz[:10], "size_median": float(np.median(sz)) if sz else 0,
              "largest_share": round(sz[0] / len(paper_ids), 3) if sz else 0,
              "weight_inside_communities_by_source": inside,
              "agreement_with_openalex": {"subfield_nmi": round(nmi_sf, 3), "subfield_purity": round(pur_sf, 3),
                                          "topic_nmi": round(nmi_tp, 3), "topic_purity": round(pur_tp, 3)},
              "examples": [{"id": r["source_id"], "n": r["n_papers"], "keywords": r["keywords"][:5],
                            "top_topics": r["top_topics"][:2]} for r in rows[:15]]}
    return rows, belongs, report


def run(cfg: Config) -> dict:
    ccfg = cfg.get("communities", {})
    weights = {**DEFAULT_WEIGHTS, **ccfg.get("weights", {})}
    levels = ccfg.get("levels") or [{"level": 0, "resolution": ccfg.get("resolution", 1.0)}]
    nodes, edges = load_processed(cfg)
    papers = nodes["Paper"]
    papers = papers[papers["name"].notna()].reset_index(drop=True)
    paper_ids = papers["node_id"].tolist()

    total, by_src = project(nodes, edges, weights, ccfg.get("max_hub_papers", 200), ccfg.get("hub_normalize", True))
    log.info("communities: %d papers, %d projected edges; weight by source: %s", len(paper_ids), len(total),
             {k: round(sum(v.values())) for k, v in by_src.items()})
    all_rows, all_belongs, reports = [], [], []
    for lv in levels:
        rows, belongs, rep = _level(lv["level"], lv["resolution"], paper_ids, papers, total, by_src, nodes,
                                    edges, ccfg)
        all_rows += rows
        all_belongs += belongs
        reports.append(rep)

    comm_nodes = pd.DataFrame(all_rows).reindex(columns=list(schema.NODE_BASE_COLUMNS) + list(
        schema.NODE_TYPES["Community"].properties) + ["top_topics", "median_year"])
    out = cfg.paths.processed
    write_parquet(comm_nodes, out / "communities.parquet")
    write_parquet(pd.DataFrame(all_belongs, columns=list(schema.EDGE_COLUMNS)), out / "edges_comm.parquet")
    report = {"projected_edges": len(total),
              "weight_by_source": {k: round(sum(v.values()), 1) for k, v in by_src.items()}, "levels": reports}
    write_json(out / "communities_report.json", report)
    for r in reports:
        log.info("communities: %s", {k: v for k, v in r.items() if k != "examples"})
    return report
