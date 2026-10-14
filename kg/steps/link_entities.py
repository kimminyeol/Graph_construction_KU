"""Step `link`: SYNONYM_OF edges between extracted entities that are the same thing (see kg/linking.py).

  processed/<run>/edges_link.parquet       SYNONYM_OF alias -> canonical (props: tier t1|t2|t3, why)
  processed/<run>/entity_clusters.parquet  node_id -> canonical_id, tier, cluster size, papers in the cluster
  processed/<run>/link_report.json         counts per tier, before/after node numbers, biggest clusters

Nodes are not merged (provenance stays); queries that want the merged view follow SYNONYM_OF / use entity_clusters.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

import numpy as np
import pandas as pd

from .. import linking, schema
from ..config import Config
from ..io import read_parquet, write_json, write_parquet
from ..linking_llm import make_judge
from .similar_papers import cached_embeddings

log = logging.getLogger(__name__)


def make_embed(cfg: Config):
    ecfg = cfg.get("embedding", {})
    return lambda texts: cached_embeddings(cfg, texts, ecfg)


def run(cfg: Config) -> dict:
    out = cfg.paths.processed
    if not (out / "edges_extract.parquet").exists():
        log.info("link: no extraction output; skipped")
        return {"skipped": True}
    params = dict(cfg.get("link", {}))
    ctx = linking.load_ctx(out)
    papers = read_parquet(out / "papers.parquet")
    papers = papers[papers["name"].notna() & papers["abstract"].notna()]
    ctx.acronyms = linking.build_acronyms(papers)
    ctx.titles = dict(zip(papers["node_id"], papers["name"]))
    t3 = "t3" in params.get("tiers", linking.DEFAULTS["tiers"])
    res = linking.compute_links(ctx, params, make_embed(cfg) if t3 else None, make_judge(cfg, cfg.get("link_judge", {})) if t3 and cfg.get("link_judge", {}).get("enabled", True) else None)
    cl, name_of, typ_of, npap = res["clusters"], res["name_of"], res["typ_of"], res["npap"]

    rows, table = [], []
    csize = Counter(c["canonical"] for c in cl.values())
    cpapers = {}
    for nid, c in cl.items():
        cpapers.setdefault(c["canonical"], set()).update(ctx.papers_of.get(nid, set()))
    for nid, c in cl.items():
        table.append({"node_id": nid, "node_type": typ_of[nid].title(),
                      "canonical_id": c["canonical"], "tier": c["tier"], "cluster_size": csize[c["canonical"]],
                      "cluster_papers": len(cpapers[c["canonical"]]), "node_papers": int(npap[nid])})
        if nid != c["canonical"]:
            tn = typ_of[nid].title()
            rows.append({"src_id": nid, "rel": "SYNONYM_OF", "dst_id": c["canonical"], "src_type": tn, "dst_type": tn,
                         "weight": round(c["score"], 4), "source": f"link:{c['tier']}",
                         "props": json.dumps({"similarity": round(c["score"], 4), "method": c["tier"], "why": c["why"]}, ensure_ascii=False)})
    edges = pd.DataFrame(rows, columns=list(schema.EDGE_COLUMNS))
    tab = pd.DataFrame(table)
    write_parquet(edges, out / "edges_link.parquet")
    write_parquet(tab, out / "entity_clusters.parquet")

    nodes = ctx.nodes
    before = {t: int((nodes.typ == t).sum()) for t in linking.TYPES}
    canon = tab[tab.node_id == tab.canonical_id]
    after = {t: int((canon.node_type.str.lower() == t).sum()) for t in linking.TYPES}
    single = lambda df, col: round(float((df[col] == 1).mean()), 3)
    report = {"params": {**linking.DEFAULTS, **params}, **res["stats"], "alias_edges": len(edges), "alias_edges_by_tier": edges["source"].value_counts().to_dict(),
              "nodes_before": before, "nodes_after": after,
              "single_paper_share_before": single(tab, "node_papers"), "single_paper_share_after": single(canon, "cluster_papers"),
              "biggest_clusters": [f"{name_of[c]} ({n} nodes, {len(cpapers[c])} papers)" for c, n in csize.most_common(15)]}
    write_json(out / "link_report.json", report)
    log.info("link: %s", {k: v for k, v in report.items() if k not in ("params", "biggest_clusters")})
    return {"alias_edges": len(edges), "nodes_before": before, "nodes_after": after}
