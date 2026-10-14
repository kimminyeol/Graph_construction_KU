"""Step `s2_refs`: add CITES edges from Semantic Scholar reference lists.

OpenAlex has no reference list for most arXiv preprints, so those papers have no outgoing CITES
edges. For corpus papers without an OpenAlex reference list we query Semantic Scholar by arXiv id
(else DOI), match the returned references to corpus papers by arXiv id / DOI, and write the new
edges (those the base graph does not already have) to processed/<run>/edges_s2.parquet.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

from .. import schema
from ..config import Config
from ..io import read_parquet, write_json, write_parquet
from ..sources.openalex import short_id
from ..sources.semanticscholar import S2References
from .build_base_graph import load_alt_ids, load_corpus_works
from .verify_metadata import arxiv_id, plain_doi

log = logging.getLogger(__name__)
SOURCE = "semanticscholar"


def _arxiv_key(a: str | None) -> str | None:
    """'2307.09288v2' -> '2307.09288'; old-style 'cs/0112017' kept as is."""
    return re.sub(r"v\d+$", "", a.lower()) if a else None


def run(cfg: Config) -> dict:
    scfg = cfg.get("semanticscholar", {})
    papers = read_parquet(cfg.paths.processed / "papers.parquet")
    papers = papers[papers["name"].notna()]
    works, _ = load_corpus_works(cfg, include_alt=True)
    to_canonical = {a: c for c, alts in load_alt_ids(cfg).items() for a in alts}
    canon_ids = set(papers["source_id"])

    # every identifier (own + merged versions') -> canonical paper id
    by_arxiv: dict[str, str] = {}
    by_doi: dict[str, str] = {}
    own_keys: dict[str, list[str]] = {}
    has_oa_refs: set[str] = set()
    for w in works:
        wid = short_id(w["id"])
        c = to_canonical.get(wid, wid)
        if c not in canon_ids:
            continue
        a, d = arxiv_id(w), plain_doi(w)
        if a:
            by_arxiv[a] = c
            own_keys.setdefault(c, []).append(f"ARXIV:{a}")
        if d:
            by_doi[d] = c
            own_keys.setdefault(c, []).append(f"DOI:{d}")
        if w.get("doi") and not a and not d:
            by_doi[w["doi"].replace("https://doi.org/", "").lower()] = c
        if c == wid and (w.get("referenced_works") or []):
            has_oa_refs.add(c)

    only_missing = scfg.get("only_missing", True)
    todo = {c: keys[0] for c, keys in own_keys.items() if not (only_missing and c in has_oa_refs)}
    limit = scfg.get("max_papers")
    if limit:
        todo = dict(list(todo.items())[:limit])
    log.info("s2_refs: %d corpus papers, %d have OpenAlex references, querying %d",
             len(canon_ids), len(has_oa_refs), len(todo))

    got = S2References(cfg.paths.raw / "semanticscholar").lookup(todo.values())

    base = read_parquet(cfg.paths.processed / "edges.parquet")
    cites = base[base.rel == "CITES"]
    existing = set(zip(cites.src_id, cites.dst_id))
    edges: dict[tuple[str, str], dict] = {}
    n_found = n_with_refs = n_listed = n_matched = 0
    for c, key in todo.items():
        rec = got.get(key)
        if not rec or not rec.get("found"):
            continue
        n_found += 1
        n_with_refs += rec["n_listed"] > 0
        n_listed += len(rec["refs"])
        src = schema.make_id("Paper", c)
        for r in rec["refs"]:
            dst_c = (by_arxiv.get(_arxiv_key(r["arxiv"])) if r["arxiv"] else None) or \
                    (by_doi.get(r["doi"]) if r["doi"] else None)
            if not dst_c or dst_c == c:
                continue
            dst = schema.make_id("Paper", dst_c)
            n_matched += 1
            if (src, dst) not in existing and (src, dst) not in edges:
                edges[(src, dst)] = {"src_id": src, "rel": "CITES", "dst_id": dst, "src_type": "Paper",
                                     "dst_type": "Paper", "weight": 1.0, "source": SOURCE, "props": "{}"}
    edge_df = pd.DataFrame(list(edges.values()), columns=list(schema.EDGE_COLUMNS))
    write_parquet(edge_df, cfg.paths.processed / "edges_s2.parquet")

    new_per_paper = edge_df.src_id.value_counts() if len(edge_df) else pd.Series(dtype=int)
    report = {"corpus_papers": len(canon_ids), "queried": len(todo), "found_in_s2": n_found,
              "found_with_reference_list": int(n_with_refs),
              "coverage_found": round(n_found / max(1, len(todo)), 3),
              "coverage_with_refs": round(n_with_refs / max(1, len(todo)), 3),
              "refs_with_ids_total": n_listed, "refs_matched_to_corpus": n_matched,
              "new_cites_edges": len(edge_df), "papers_gaining_edges": int(new_per_paper.shape[0]),
              "base_cites_edges": len(existing),
              "mean_new_edges_per_gaining_paper": round(float(new_per_paper.mean()), 2) if len(new_per_paper) else 0}
    write_json(cfg.paths.processed / "s2_refs_report.json", report)
    log.info("s2_refs: %s", report)
    return report
