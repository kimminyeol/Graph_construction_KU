"""Step `arxiv_refs`: CITES edges from arXiv HTML reference lists (rule-based, no LLM).

OpenAlex has no reference list for most arXiv preprints and Semantic Scholar misses some. For corpus
papers that have an arXiv id and no OpenAlex reference list we read the bibliography of their arXiv HTML
page and match every entry to a corpus paper:

  id            an arXiv id or DOI in the entry (text or link) is the id of a corpus paper
  title         the corpus paper's normalised title occurs in the entry; for short titles (<= 5 words) the first
                author's surname must also occur in the entry
  title_fuzzy   >= 90% of the distinctive title words (titles of >= 6 words) occur in the entry AND the first
                author's surname occurs
and the year in the entry must be compatible with the paper's year. Self-citations are dropped.

`loose_candidates` is a deliberately permissive retriever used only to prepare gold-standard labels.
Output: processed/<run>/edges_arxivref.parquet, arxiv_refs_report.json
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

from .. import schema
from ..config import Config
from ..io import read_parquet, write_json, write_parquet
from ..sources.arxiv_html import ArxivBibliographies, find_ids
from ..sources.openalex import short_id
from .build_base_graph import load_alt_ids, load_corpus_works
from .verify_metadata import arxiv_id

log = logging.getLogger(__name__)
SOURCE = "arxiv_html"
_YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")


def norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", s))


def surname(display_name: str | None) -> str | None:
    if not display_name:
        return None
    n = display_name.strip()
    part = n.split(",")[0] if "," in n else n.split()[-1]
    w = norm(part).split()
    return w[-1] if w else None


def distinct_tokens(title: str) -> list[str]:
    return [t for t in dict.fromkeys(norm(title).split()) if len(t) > 2]


class CorpusIndex:
    """Lookup structures over corpus papers (canonical + merged-version records)."""

    def __init__(self, papers: pd.DataFrame, works: dict[str, dict], alt_ids: dict[str, list[str]]):
        self.pid_year: dict[str, int] = {}
        self.by_arxiv: dict[str, str] = {}
        self.by_doi: dict[str, str] = {}
        self.titles: dict[str, list[str]] = {}      # pid -> normalised titles (canonical + versions)
        self.surname: dict[str, str | None] = {}
        self.raw_ids: dict[str, list[str]] = {}
        for sid, name, year in zip(papers["source_id"], papers["name"], papers["year"]):
            if not name:
                continue
            pid = sid
            raws = [sid] + list(alt_ids.get(sid, []))
            self.raw_ids[pid] = raws
            self.pid_year[pid] = int(year) if pd.notna(year) else 0
            tl = [norm(name)]
            for rid in raws:
                w = works.get(rid)
                if not w:
                    continue
                if w.get("title"):
                    tl.append(norm(w["title"]))
                d = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                if d:
                    self.by_doi.setdefault(d, pid)
                if (a := arxiv_id(w)):
                    self.by_arxiv.setdefault(a, pid)
                    self.by_doi.setdefault(f"10.48550/arxiv.{a}", pid)
            self.titles[pid] = [t for t in dict.fromkeys(tl) if len(t.split()) >= 3]
            w0 = works.get(sid) or {}
            auth = (w0.get("authorships") or [{}])[0] if w0.get("authorships") else {}
            self.surname[pid] = surname((auth.get("author") or {}).get("display_name") or auth.get("raw_author_name"))
        df = Counter(t for ts in self.titles.values() for t in {w for key in ts for w in key.split()})
        self.df = df
        self.post: dict[str, list[tuple[str, str]]] = defaultdict(list)   # rarest token -> (pid, title key)
        for pid, ts in self.titles.items():
            for key in ts:
                toks = [w for w in key.split() if len(w) > 2] or key.split()
                rare = min(toks, key=lambda w: (df[w], -len(w)))
                self.post[rare].append((pid, key))
        self.n_titles = sum(len(v) for v in self.titles.values())
        self.tok_post: dict[str, set[str]] = defaultdict(set)   # any token with df <= 400 -> pids (permissive retrieval)
        for pid, ts in self.titles.items():
            for key in ts:
                for w in set(key.split()):
                    if len(w) > 2 and df[w] <= 400:
                        self.tok_post[w].add(pid)

    def idf(self, tok: str) -> float:
        return math.log((self.n_titles + 1) / (self.df.get(tok, 0) + 1))


def year_ok(entry_years: list[int], paper_year: int) -> bool:
    if not entry_years or not paper_year:
        return True
    return any(y - paper_year <= 1 and paper_year - y <= 4 for y in entry_years)


def match_entry(entry: dict, idx: CorpusIndex, self_pid: str | None = None) -> list[dict]:
    """Corpus papers cited by one bibliography entry (strict, automatic rules)."""
    text = norm(entry["text"])
    padded = f" {text} "
    toks = set(text.split())
    years = [int(y) for y in _YEAR.findall(entry["text"])]
    out: dict[str, dict] = {}
    # 1. identifiers
    arx, dois = entry.get("arxiv_ids") or [], entry.get("dois") or []
    for a in arx:
        if (p := idx.by_arxiv.get(a.lower())) and p != self_pid:
            out[p] = {"pid": p, "match": "id", "detail": f"arxiv:{a}"}
    for d in dois:
        if (p := idx.by_doi.get(d.lower())) and p != self_pid and p not in out:
            out[p] = {"pid": p, "match": "id", "detail": f"doi:{d}"}
    if out:
        return list(out.values())
    # 2. titles
    best: dict[str, tuple[int, str, str]] = {}
    seen_rare = [t for t in toks if t in idx.post]
    for rare in seen_rare:
        for pid, key in idx.post[rare]:
            if pid == self_pid or pid in best and len(best[pid][1].split()) >= len(key.split()):
                continue
            n = len(key.split())
            sn = idx.surname.get(pid)
            has_sn = bool(sn and f" {sn} " in padded)
            if not year_ok(years, idx.pid_year.get(pid, 0)):
                continue
            if f" {key} " in padded and (n >= 6 or has_sn):
                best[pid] = (2, key, "title")
                continue
            tt = distinct_tokens(key)
            if n >= 6 and len(tt) >= 5 and has_sn and sum(t in toks for t in tt) / len(tt) >= 0.9:
                if pid not in best:
                    best[pid] = (1, key, "title_fuzzy")
    if not best:
        return []
    top = max(v[0] for v in best.values())
    cands = {p: v for p, v in best.items() if v[0] == top}
    longest = max(len(v[1].split()) for v in cands.values())
    return [{"pid": p, "match": v[2], "detail": v[1][:80]} for p, v in cands.items() if len(v[1].split()) == longest]


def loose_candidates(entry: dict, idx: CorpusIndex, self_pid: str | None = None, top: int = 3) -> list[dict]:
    """Permissive retrieval for gold labelling: ids, weighted title-word overlap >= 0.5, or surname + 2 title words."""
    strict = {m["pid"]: m for m in match_entry(entry, idx, self_pid)}
    text = norm(entry["text"])
    toks = set(text.split())
    cand_pids: set[str] = set()
    for t in toks:
        cand_pids |= idx.tok_post.get(t, set())
    out = []
    for pid in cand_pids | set(strict):
        if pid == self_pid:
            continue
        best = (0.0, "")
        for key in idx.titles.get(pid, []):
            tt = distinct_tokens(key)
            if not tt:
                continue
            tot = sum(idx.idf(t) for t in tt)
            got = sum(idx.idf(t) for t in tt if t in toks)
            cov = got / tot if tot else 0
            if cov > best[0]:
                best = (cov, key)
        sn = idx.surname.get(pid)
        has_sn = bool(sn and sn in toks)
        if pid in strict or best[0] >= 0.5 or (has_sn and best[0] >= 0.3):
            out.append({"pid": pid, "score": round(best[0] + (0.2 if has_sn else 0) + (1 if pid in strict else 0), 3),
                        "title": best[1], "strict": pid in strict})
    out.sort(key=lambda x: -x["score"])
    return out[:top]


def build_index(cfg: Config) -> tuple[CorpusIndex, dict[str, dict], pd.DataFrame]:
    papers = read_parquet(cfg.paths.processed / "papers.parquet")
    papers = papers[papers["name"].notna()]
    works_list, _ = load_corpus_works(cfg, include_alt=True)
    works = {short_id(w["id"]): w for w in works_list}
    return CorpusIndex(papers, works, load_alt_ids(cfg)), works, papers


def run(cfg: Config) -> dict:
    acfg = cfg.get("arxiv_refs", {})
    idx, works, papers = build_index(cfg)
    alt = load_alt_ids(cfg)
    targets = []
    for sid, refs in zip(papers["source_id"], papers["referenced_works_count"]):
        arx = next((a for rid in [sid] + alt.get(sid, []) if (w := works.get(rid)) and (a := arxiv_id(w))), None)
        if arx and (not acfg.get("only_missing", True) or not (refs or 0)):
            targets.append((sid, arx))
    if acfg.get("max_papers"):
        targets = targets[: acfg["max_papers"]]
    log.info("arxiv_refs: %d target papers", len(targets))
    bib = ArxivBibliographies(cfg.paths.raw / "arxiv_html", min_interval_s=acfg.get("min_interval_s", 0.6))

    base = read_parquet(cfg.paths.processed / "edges.parquet")
    existing = set(zip(base.loc[base.rel == "CITES", "src_id"], base.loc[base.rel == "CITES", "dst_id"]))
    for extra in sorted(cfg.paths.processed.glob("edges_s2.parquet")):
        e = pd.read_parquet(extra)
        existing |= set(zip(e.src_id, e.dst_id))
    edges: dict[tuple[str, str], dict] = {}
    found = n_entries = n_matched = 0
    by_match = Counter()
    sizes = []
    for i, (sid, arx) in enumerate(targets, 1):
        rec = bib.get(arx)
        if not rec["found"]:
            continue
        found += 1
        sizes.append(len(rec["entries"]))
        src = schema.make_id("Paper", sid)
        for ent in rec["entries"]:
            n_entries += 1
            for m in match_entry(ent, idx, self_pid=sid):
                n_matched += 1
                by_match[m["match"]] += 1
                dst = schema.make_id("Paper", m["pid"])
                if (src, dst) in existing or (src, dst) in edges:
                    continue
                edges[(src, dst)] = {"src_id": src, "rel": "CITES", "dst_id": dst, "src_type": "Paper", "dst_type": "Paper",
                                     "weight": 1.0, "source": SOURCE,
                                     "props": json.dumps({"match": m["match"], "entry": ent["text"][:220], "idx": ent["idx"]}, ensure_ascii=False)}
        if i % 200 == 0:
            log.info("arxiv_refs: %d/%d papers, %d new edges, %d fetched", i, len(targets), len(edges), bib.fetched)
    edge_df = pd.DataFrame(list(edges.values()), columns=list(schema.EDGE_COLUMNS))
    write_parquet(edge_df, cfg.paths.processed / "edges_arxivref.parquet")
    per = edge_df.src_id.value_counts() if len(edge_df) else pd.Series(dtype=int)
    report = {"target_papers": len(targets), "html_found": found, "html_found_rate": round(found / max(1, len(targets)), 3),
              "median_entries": float(pd.Series(sizes).median()) if sizes else 0, "entries": n_entries,
              "matched_entries": n_matched, "matches_by_rule": dict(by_match), "new_cites_edges": len(edge_df),
              "papers_gaining_edges": int(per.shape[0]), "mean_new_edges_per_gaining_paper": round(float(per.mean()), 2) if len(per) else 0}
    write_json(cfg.paths.processed / "arxiv_refs_report.json", report)
    log.info("arxiv_refs: %s", report)
    return report
