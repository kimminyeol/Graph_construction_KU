"""Step `select`: choose the final corpus from the raw candidate pool.

Candidates that failed `verify` (corrupt / out-of-scope records) are skipped, and
preprint/published versions of the same paper are merged (see merge_versions).
Pass 1: every unit contributes up to its quota (in its citation rank order).
Pass 2: leftover capacity is filled round-robin across units.
A paper matched by several units is kept once, remembering all its units.
Output: processed/<run>/corpus.parquet (paper_id, unit, rank, corpus_units).
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from itertools import islice
from typing import Iterator

import pandas as pd

from ..config import Config
from ..io import iter_jsonl, write_json, write_parquet
from ..sources.openalex import short_id
from .collect_openalex import plan_units

log = logging.getLogger(__name__)


def passes_local_filters(work: dict, lf: dict) -> tuple[bool, str]:
    if short_id(work["id"]) in set(lf.get("exclude_ids") or []):
        return False, "excluded_id"
    venue = (((work.get("primary_location") or {}).get("source") or {}).get("id"))
    if venue and short_id(venue) in set(lf.get("exclude_venues") or []):
        return False, "excluded_venue"
    doi = (work.get("doi") or "").replace("https://doi.org/", "").lower()
    if doi and doi.startswith(tuple(lf.get("exclude_doi_prefixes") or [])):
        return False, "excluded_doi_prefix"
    if not (work.get("title") or work.get("display_name")):
        return False, "no_title"
    inv = work.get("abstract_inverted_index") or {}
    n_words = sum(len(v) for v in inv.values())
    if n_words < lf.get("min_abstract_words", 0):
        return False, "abstract_too_short"
    if n_words > lf.get("max_abstract_words", 10**9):
        return False, "abstract_too_long"  # usually full text / front matter dumped into abstract
    if work.get("is_paratext") or work.get("is_retracted"):
        return False, "paratext_or_retracted"
    return True, ""


def iter_candidates(cfg: Config, rejected: dict[str, int] | None = None) -> Iterator[tuple[str, dict]]:
    """(unit key, work) for every collected work of the current plan that passes local filters,
    in rank order within each unit. A work can appear under several units.

    Only the first quota x oversample records of each unit are read: the raw store is shared, and a
    larger run may have fetched more, which must not change a smaller run's corpus."""
    works_dir = cfg.paths.openalex_raw / "works"
    lf = cfg["corpus"].get("local_filters", {})
    oversample = cfg["corpus"].get("oversample", 1.5)
    for u in plan_units(cfg):
        seen = set()
        for w in islice(iter_jsonl(works_dir / f"{u.key}.jsonl"), math.ceil(u.quota * oversample)):
            pid = short_id(w["id"])
            if pid in seen:
                continue
            seen.add(pid)
            ok, why = passes_local_filters(w, lf)
            if ok:
                yield u.key, w
            elif rejected is not None:
                rejected[why] += 1


def _title_key(title: str | None) -> str | None:
    words = re.findall(r"[a-z0-9]+", (title or "").lower())
    key = "".join(words)
    return key if len(words) >= 4 and len(key) >= 20 else None  # short titles are too generic


def merge_versions(meta: dict[str, dict], mcfg: dict) -> dict[str, str]:
    """Map every candidate to a canonical id. OpenAlex keeps the arXiv preprint and the
    published version of a paper as separate works; same normalised title within a few years
    -> one paper. Canonical = has a reference list > is an article > more citations."""
    canon = {pid: pid for pid in meta}
    if not mcfg.get("enabled", True):
        return canon
    max_gap = mcfg.get("max_year_gap", 3)
    groups: dict[str, list[str]] = defaultdict(list)
    for pid, m in meta.items():
        if (k := _title_key(m["title"])):
            groups[k].append(pid)
    for pids in groups.values():
        if len(pids) < 2:
            continue
        pids.sort(key=lambda p: (meta[p]["has_refs"], meta[p]["is_article"], meta[p]["cited"]), reverse=True)
        best = pids[0]
        for p in pids[1:]:
            y0, y1 = meta[best]["year"], meta[p]["year"]
            if y0 is None or y1 is None or abs(y0 - y1) <= max_gap:
                canon[p] = best
    return canon


def merge_versions_v2(meta: dict[str, dict], mcfg: dict, vecs: dict[str, "np.ndarray"] | None = None) -> tuple[dict[str, str], dict]:
    """Version merging with author guards (used by `corpus.merge_versions.v2`).

    R1  same normalised title (>= 4 words), year gap <= max_year_gap, AND shared authors (overlap >= title_author_overlap
        when both records list authors). v1 had no author check and merged two different papers with the same title.
    R2  near-identical content but a different title (arXiv title vs. journal title): embedding cosine >= embedding_cos,
        author overlap >= author_overlap, year gap <= max_year_gap, and the pair is one preprint + one non-preprint
        (a preprint and its published version). Preprint/preprint or article/article pairs are usually successors
        (Qwen2 / Qwen2.5, PaLI / PaLI-X) and need author overlap >= same_type_author_overlap and year gap <= 1.
    R3  identical abstract (first 300 normalised characters, >= 150) under a different title, author overlap >= title_author_overlap
        (when both list authors), year gap <= max_year_gap: preprint and journal versions renamed on publication (VioLA, AWQ, PeFoMed).
        Publisher boilerplate is shared too, but `abstracts` has already removed it.
    Canonical record of a group: has reference list > is an article > more citations."""
    import numpy as np

    from .verify_metadata import author_overlap
    ids = list(meta)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    stats = {"r1_title": 0, "r2_embedding": 0, "r1_blocked_by_authors": 0, "r3_same_abstract": 0, "r3_blocked_by_authors": 0}
    gap_max = mcfg.get("max_year_gap", 3)

    def ao(a: str, b: str) -> float | None:
        return author_overlap(meta[a]["authors"], meta[b]["authors"]) if meta[a]["authors"] and meta[b]["authors"] else None

    def gap(a: str, b: str) -> int:
        ya, yb = meta[a]["year"], meta[b]["year"]
        return 0 if ya is None or yb is None else abs(ya - yb)
    groups: dict[str, list[str]] = defaultdict(list)
    for pid, m in meta.items():
        if (k := _title_key(m["title"])):
            groups[k].append(pid)
    for pids in groups.values():
        for i, a in enumerate(pids):
            for b in pids[i + 1:]:
                if gap(a, b) > gap_max:
                    continue
                o = ao(a, b)
                if o is not None and o < mcfg.get("title_author_overlap", 0.5):
                    stats["r1_blocked_by_authors"] += 1
                    continue
                if find(a) != find(b):
                    parent[find(a)] = find(b)
                    stats["r1_title"] += 1
    if mcfg.get("same_abstract", False):
        by_abs: dict[str, list[str]] = defaultdict(list)
        for pid, m in meta.items():
            k = " ".join(re.findall(r"[a-z0-9]+", (m.get("abstract") or "").lower()))[:300]
            if len(k) >= 150:
                by_abs[k].append(pid)
        for pids in by_abs.values():
            for i, a in enumerate(pids):
                for b in pids[i + 1:]:
                    if find(a) == find(b) or gap(a, b) > gap_max:
                        continue
                    o = ao(a, b)
                    if o is not None and o < mcfg.get("title_author_overlap", 0.5):
                        stats["r3_blocked_by_authors"] += 1
                        continue
                    parent[find(a)] = find(b)
                    stats["r3_same_abstract"] += 1
    if vecs:
        M = np.stack([vecs[i] for i in ids])
        cos_min = mcfg.get("embedding_cos", 0.95)
        for st in range(0, len(ids), 2048):
            sims = M[st:st + 2048] @ M.T
            for r, c in zip(*np.where(sims >= cos_min)):
                i, j = st + int(r), int(c)
                if j <= i:
                    continue
                a, b = ids[i], ids[j]
                if find(a) == find(b) or gap(a, b) > gap_max:
                    continue
                o = ao(a, b)
                if o is None:
                    continue
                prep_a, prep_b = meta[a]["is_preprint"], meta[b]["is_preprint"]
                if prep_a != prep_b:
                    ok = o >= mcfg.get("author_overlap", 0.6)
                else:
                    ok = o >= mcfg.get("same_type_author_overlap", 0.95) and gap(a, b) <= 1
                if ok:
                    parent[find(a)] = find(b)
                    stats["r2_embedding"] += 1
    comps: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        comps[find(i)].append(i)
    canon = {}
    for members in comps.values():
        members.sort(key=lambda p: (meta[p]["has_refs"], meta[p]["is_article"], meta[p]["cited"]), reverse=True)
        for m in members:
            canon[m] = members[0]
    return canon, stats


def load_pool_works(cfg: Config, wanted: set[str]) -> dict[str, dict]:
    works: dict[str, dict] = {}
    for u in plan_units(cfg):
        for w in iter_jsonl(cfg.paths.openalex_raw / "works" / f"{u.key}.jsonl"):
            wid = short_id(w["id"])
            if wid in wanted and wid not in works:
                works[wid] = w
    missing = wanted - set(works)
    if missing:
        raise RuntimeError(f"{len(missing)} pool works missing from the raw store, e.g. {next(iter(missing))}")
    return works


def run_fixed(cfg: Config) -> dict:
    """`corpus.fixed_from: <run>`: take exactly the works of another run's corpus (its papers and their merged versions),
    and re-derive the version merging with the current rules. The set of underlying works stays identical, so before/after
    comparisons measure the new rules, not a different sample."""
    from ..config import PROJECT_ROOT
    from .build_base_graph import reconstruct_abstract, clean_text
    from .similar_papers import cached_embeddings, paper_text
    base_run = cfg["corpus"]["fixed_from"]
    base = read_corpus(PROJECT_ROOT / "data" / "processed" / base_run / "corpus.parquet")
    pool = list(base["paper_id"]) + [a for alts in base["alt_ids"] for a in alts]
    works = load_pool_works(cfg, set(pool))
    ver = load_verification(cfg)
    repaired = {r.paper_id: (r.new_title, r.new_abstract) for r in ver.itertuples() if r.status == "repaired"} if ver is not None else {}
    fixes = load_abstract_fixes(cfg)
    meta, texts = {}, []
    for pid in pool:
        w = works[pid]
        title, abstract = repaired.get(pid) or (w.get("title") or w.get("display_name"), reconstruct_abstract(w.get("abstract_inverted_index")))
        if pid in fixes:
            abstract = fixes[pid][1]
        meta[pid] = {"title": title, "abstract": abstract, "year": w.get("publication_year"), "has_refs": (w.get("referenced_works_count") or 0) > 0,
                     "is_article": w.get("type") == "article", "is_preprint": w.get("type") == "preprint", "cited": w.get("cited_by_count") or 0,
                     "authors": [(a.get("author") or {}).get("display_name") or a.get("raw_author_name") or "" for a in w.get("authorships") or []]}
        texts.append(paper_text(clean_text(title), abstract))
    mcfg = cfg["corpus"].get("merge_versions", {})
    vecs = None
    if mcfg.get("v2"):
        emb = cached_embeddings(cfg, texts, cfg.get("embedding", {}))
        vecs = dict(zip(pool, emb))
    canon, stats = merge_versions_v2(meta, mcfg, vecs)
    alt_ids: dict[str, list[str]] = defaultdict(list)
    for pid, c in canon.items():
        if pid != c:
            alt_ids[c].append(pid)
    base_row = {r.paper_id: r for r in base.itertuples()}
    member_row = {a: r for r in base.itertuples() for a in list(r.alt_ids)}
    rows = []
    for c in dict.fromkeys(canon.values()):
        src = base_row.get(c) or member_row.get(c) or next((base_row[m] for m in alt_ids[c] if m in base_row), None)
        rows.append({"paper_id": c, "unit": src.unit if src is not None else "merged", "rank": int(src.rank) if src is not None else 0,
                     "corpus_units": list(src.corpus_units) if src is not None else [], "alt_ids": sorted(alt_ids.get(c, []))})
    corpus = pd.DataFrame(rows)
    write_parquet(corpus, cfg.paths.processed / "corpus.parquet")
    report = {"fixed_from": base_run, "pool_works": len(pool), "papers": len(corpus), "versions_merged": int(sum(len(v) for v in alt_ids.values())),
              "merge_rule_counts": stats, "baseline_papers": len(base)}
    write_json(cfg.paths.processed / "select_report.json", report)
    log.info("select (fixed from %s): %s", base_run, report)
    return report


def read_corpus(path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_abstract_fixes(cfg: Config) -> dict[str, tuple[str, str | None]]:
    """paper id -> (action, new abstract) for the records the `abstracts` step replaced or dropped; {} when it was not run."""
    path = cfg.paths.processed / "abstract_check.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    df = df[df.action.isin(["replaced", "dropped"])]
    return {r.paper_id: (r.action, r.new_abstract if r.action == "replaced" else None) for r in df.itertuples()}


def load_verification(cfg: Config) -> pd.DataFrame | None:
    path = cfg.paths.processed / "verification.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if cfg.get("verify", {}).get("enabled", True):
        raise FileNotFoundError(f"{path} missing: run the `verify` step first (or set verify.enabled: false)")
    return None


def run(cfg: Config) -> dict:
    if cfg["corpus"].get("fixed_from"):
        return run_fixed(cfg)
    max_papers = cfg["corpus"]["max_papers"]
    units = plan_units(cfg)
    ver = load_verification(cfg)
    dropped = set(ver.loc[ver.status.str.startswith("drop"), "paper_id"]) if ver is not None else set()
    repaired = ({r.paper_id: r.new_title for r in ver.itertuples() if r.status == "repaired"}
                if ver is not None else {})

    seq: list[tuple[str, str]] = []
    meta: dict[str, dict] = {}
    rejected: dict[str, int] = defaultdict(int)
    for unit_key, w in iter_candidates(cfg, rejected):
        pid = short_id(w["id"])
        if pid in dropped:
            rejected["failed_verification"] += 1
            continue
        seq.append((unit_key, pid))
        meta.setdefault(pid, {"title": repaired.get(pid) or w.get("title") or w.get("display_name"),
                              "year": w.get("publication_year"),
                              "has_refs": (w.get("referenced_works_count") or 0) > 0,
                              "is_article": w.get("type") == "article",
                              "cited": w.get("cited_by_count") or 0})
    if ver is not None and (unchecked := set(meta) - set(ver["paper_id"])):
        raise RuntimeError(f"{len(unchecked)} candidates were never verified (raw data changed since `verify`); "
                           f"run with --from verify")
    canon = merge_versions(meta, cfg["corpus"].get("merge_versions", {}))
    alt_ids: dict[str, list[str]] = defaultdict(list)
    for pid, c in canon.items():
        if pid != c:
            alt_ids[c].append(pid)

    candidates: dict[str, list[str]] = {u.key: [] for u in units}  # unit -> ordered unique canonical ids
    matched_units: dict[str, list[str]] = defaultdict(list)
    for unit_key, pid in seq:
        c = canon[pid]
        if c not in matched_units or unit_key not in matched_units[c]:
            matched_units[c].append(unit_key)
        if c not in candidates[unit_key]:
            candidates[unit_key].append(c)
    for key, ids in candidates.items():
        if not ids:
            log.warning("unit %s has no candidates (not collected yet?)", key)

    chosen: dict[str, tuple[str, int]] = {}  # paper -> (unit that selected it, rank in that unit)
    cursors = {u.key: 0 for u in units}

    def take_next(unit_key: str) -> bool:
        ids = candidates[unit_key]
        while cursors[unit_key] < len(ids):
            pid = ids[cursors[unit_key]]
            cursors[unit_key] += 1
            if pid not in chosen:
                chosen[pid] = (unit_key, cursors[unit_key])
                return True
        return False

    for u in units:                      # pass 1: quotas
        taken = 0
        while taken < u.quota and len(chosen) < max_papers and take_next(u.key):
            taken += 1
    active = [u.key for u in units]      # pass 2: round-robin fill
    while len(chosen) < max_papers and active:
        active = [k for k in active if len(chosen) < max_papers and take_next(k)]

    corpus = pd.DataFrame([{"paper_id": pid, "unit": uk, "rank": rank, "corpus_units": matched_units[pid],
                            "alt_ids": sorted(alt_ids.get(pid, []))}
                           for pid, (uk, rank) in chosen.items()])
    write_parquet(corpus, cfg.paths.processed / "corpus.parquet")
    report = {"selected": len(corpus), "target": max_papers,
              "candidates_unique": len(matched_units), "rejected_local": dict(rejected),
              "versions_merged": int(sum(len(v) for v in alt_ids.values())),
              "selected_with_alt_ids": int((corpus["alt_ids"].map(len) > 0).sum()) if len(corpus) else 0,
              "per_source": corpus["unit"].str.split("__").str[0].value_counts().to_dict() if len(corpus) else {}}
    write_json(cfg.paths.processed / "select_report.json", report)
    log.info("select: %s", report)
    if len(corpus) < max_papers:
        log.warning("corpus has %d < %d papers: collect more (raise oversample) or add sources",
                    len(corpus), max_papers)
    return report
