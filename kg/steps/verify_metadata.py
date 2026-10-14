"""Step `verify`: check candidate records against their identifier's registry.

For each candidate (arXiv id first, else DOI via Crossref, then DataCite):
  ok              registry title matches
  doi_mismatch    DOI belongs to another paper, but arXiv has this title + authors -> kept, DOI cleared
  repaired       title differs but the author lists match -> same paper whose
                  OpenAlex title/abstract were overwritten; registry metadata is used
  drop_identity   title and authors both differ -> cannot tell which paper this is
  drop_year       verified year is outside corpus.years (e.g. a 2017 paper filed as 2025), or a
                  pre-2022 (MAG-era) OpenAlex id claims a year far past 2021 (see check_record_age)
  unverified      no identifier / unknown to the registry -> kept as is
Output: processed/<run>/verification.parquet; registry responses cached in data/raw/registry/.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from ..config import Config
from ..io import write_json, write_parquet
from ..sources.openalex import short_id
from ..sources.registries import Arxiv, Crossref, DataCite
from .build_base_graph import clean_text
from .select_corpus import iter_candidates

log = logging.getLogger(__name__)

_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$", re.I)
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(.+?)(?:v\d+)?(?:\.pdf)?/?$", re.I)


def arxiv_id(work: dict) -> str | None:
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    if (m := _ARXIV_DOI.search(doi)):
        return m.group(1).lower()
    loc = work.get("primary_location") or {}
    for url in (loc.get("landing_page_url"), loc.get("pdf_url")):
        if url and (m := _ARXIV_URL.search(url)):
            return m.group(1).lower()
    return None


def plain_doi(work: dict) -> str | None:
    doi = (work.get("doi") or "").replace("https://doi.org/", "").lower()
    return doi if doi and not _ARXIV_DOI.search(doi) else None


def _norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", s))


def title_similarity(a: str | None, b: str | None) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    short, long_ = sorted((na, nb), key=len)
    if len(short.split()) >= 3 and long_.startswith(short):  # one side drops the subtitle
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def author_overlap(a: list[str], b: list[str]) -> float | None:
    """Share of name tokens in common (robust to 'Wei, Jason' vs 'Jason Wei')."""
    ta = {t for n in a for t in _norm(n).split() if len(t) > 1}
    tb = {t for n in b for t in _norm(n).split() if len(t) > 1}
    if not ta or not tb:
        return None
    return len(ta & tb) / min(len(ta), len(tb))


def judge(work: dict, rec: dict | None, registry: str | None, vcfg: dict, years: tuple[int, int]) -> dict:
    title = clean_text(work.get("title") or work.get("display_name"))
    out = {"paper_id": short_id(work["id"]), "registry": registry, "registry_id": rec and rec["key"],
           "oa_title": title, "registry_title": None, "title_similarity": None, "author_overlap": None,
           "oa_year": work.get("publication_year"), "registry_year": None,
           "new_title": None, "new_abstract": None, "new_year": None, "legacy_id": False,
           "status": "unverified"}
    if not rec or not rec.get("found") or not rec.get("title"):
        return out
    names = [(a.get("author") or {}).get("display_name") or a.get("raw_author_name") or ""
             for a in work.get("authorships") or []]
    tsim = title_similarity(title, rec["title"])
    aov = author_overlap(names, rec.get("authors") or [])
    out.update(registry_title=rec["title"], title_similarity=round(tsim, 3), registry_year=rec.get("year"),
               author_overlap=None if aov is None else round(aov, 3))
    if tsim >= vcfg.get("title_threshold", 0.75):
        out["status"] = "ok"
    elif aov is not None and aov >= vcfg.get("author_threshold", 0.5):
        out.update(status="repaired", new_title=rec["title"], new_abstract=rec.get("abstract"))
    else:
        out["status"] = "drop_identity"
        return out
    ry, oy = rec.get("year"), work.get("publication_year")
    if ry and oy and abs(ry - oy) > vcfg.get("max_year_gap", 2):
        out["new_year"] = ry
        if not years[0] <= ry <= years[1]:
            out["status"] = "drop_year"
    return out


_NEW_ID = re.compile(r"^W[4-9]\d{9}$")


def check_record_age(row: dict, vcfg: dict) -> dict:
    """OpenAlex ids other than W4xxxxxxxxx+ were inherited from MAG, which stopped at the end of 2021.
    Such a record existed by 2021, so a claimed year far beyond that means an old paper whose year
    was overwritten (e.g. "Attention Is All You Need" re-deposited in 2025, Kaldi filed as 2024).
    Registries cannot catch these when the re-deposit itself carries the new year."""
    if row["status"].startswith("drop") or _NEW_ID.match(row["paper_id"]):
        return row
    year = row["new_year"] if row.get("new_year") else row["oa_year"]
    if year and year - vcfg.get("legacy_id_last_year", 2021) > vcfg.get("max_year_gap", 2):
        row = {**row, "status": "drop_year", "legacy_id": True}
    return row


def run(cfg: Config) -> dict:
    vcfg = cfg.get("verify", {})
    years = tuple(cfg["corpus"]["years"])
    works = {}
    for _, w in iter_candidates(cfg):
        works.setdefault(short_id(w["id"]), w)
    cache = cfg.paths.raw / "registry"
    arx_ids = {pid: arxiv_id(w) for pid, w in works.items()}
    arx = Arxiv(cache).lookup(v for v in arx_ids.values() if v)

    # DOI registries only for works without a usable arXiv record
    need_doi = {pid: plain_doi(w) for pid, w in works.items()
                if not (arx_ids[pid] and arx.get(arx_ids[pid], {}).get("found"))}
    cr = Crossref(cache).lookup(v for v in need_doi.values() if v)
    dc_keys = [d for d in need_doi.values() if d and not cr.get(d, {}).get("found")]
    dc = DataCite(cache).lookup(dc_keys) if vcfg.get("use_datacite", True) else {}

    arxiv_client = Arxiv(cache)
    rows = []
    for pid, w in works.items():
        a = arx_ids[pid]
        if a and arx.get(a, {}).get("found"):
            rows.append(judge(w, arx[a], "arxiv", vcfg, years))
            continue
        d = need_doi.get(pid)
        if d and cr.get(d, {}).get("found"):
            row = judge(w, cr[d], "crossref", vcfg, years)
        elif d and dc.get(d, {}).get("found"):
            row = judge(w, dc[d], "datacite", vcfg, years)
        else:
            row = judge(w, None, None, vcfg, years)
        if row["status"] == "drop_identity":
            # Title and DOI disagree. If arXiv has this exact title by these authors, the DOI is the
            # corrupt part (e.g. BLOOM carrying a cancer-treatment DOI): keep the paper, drop the DOI.
            for hit in arxiv_client.search_title(row["oa_title"] or ""):
                alt = judge(w, hit, "arxiv_title_search", vcfg, years)
                if alt["status"] == "ok" and (alt["author_overlap"] or 0) >= vcfg.get("author_threshold", 0.5):
                    row = {**alt, "status": "doi_mismatch"}
                    break
                if alt["status"] == "drop_year":
                    row = alt
                    break
        rows.append(row)
    rows = [check_record_age(r, vcfg) for r in rows]
    df = pd.DataFrame(rows)
    write_parquet(df, cfg.paths.processed / "verification.parquet")
    report = {"candidates": len(df), "status": df.status.value_counts().to_dict(),
              "registry": df.registry.fillna("none").value_counts().to_dict(),
              "examples": {s: df.loc[df.status == s, ["paper_id", "oa_title", "registry_title"]]
                              .head(5).to_dict("records") for s in ("repaired", "drop_identity", "drop_year")}}
    write_json(cfg.paths.processed / "verify_report.json", report)
    log.info("verify: %s", {k: report[k] for k in ("candidates", "status", "registry")})
    return report
