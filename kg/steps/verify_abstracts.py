"""Step `abstracts`: find OpenAlex records whose abstract belongs to another work, and repair or remove it.

`verify` compares titles only, so a record with the right title and a foreign abstract passes (the DPO and GPT-4 Technical Report
records carry unrelated abstracts; some journal records carry the publisher's boilerplate). Evidence, strongest first:

  registry    the arXiv / Crossref / DataCite abstract of the same identifier shares < `registry_overlap` of its words with the OpenAlex
              abstract AND is closer to the title (title-abstract embedding cosine) -> replaced by the registry abstract
  duplicate   the same abstract text under >= 2 different titles AND a poor title fit (blurbs)  } -> a replacement is looked up
  low         title-abstract cosine < `low_cos`, almost no title word in the abstract,          } (Semantic Scholar title match, must be
              title of >= 3 words, Latin script                                                  } closer to the title); none found -> removed
  flagged     cosine < `flag_cos` and little lexical overlap: reported, nothing changed

Why not act on similarity alone: names as titles ("Sasha"), short abstracts and generic introductions are common among correct records
(on the 5,410 papers with a registry abstract, cosine < 0.70 had 7% precision for a foreign abstract).
Output: processed/<run>/abstract_check.parquet, abstract_report.json. `select` and `build` apply action replaced / dropped.
"""
from __future__ import annotations

import logging
import re
from collections import Counter

import numpy as np
import pandas as pd

from ..config import Config
from ..io import write_json, write_parquet
from ..sources.openalex import short_id
from ..sources.registries import Arxiv, Crossref, DataCite, S2Title
from .build_base_graph import clean_text, reconstruct_abstract
from .select_corpus import iter_candidates, load_abstract_fixes, load_verification  # noqa: F401  (load_abstract_fixes re-exported)
from .similar_papers import cached_embeddings
from .verify_metadata import title_similarity

log = logging.getLogger(__name__)
DEFAULTS = {"registry_overlap": 0.5, "min_gain": 0.05, "low_cos": 0.55, "low_lex": 0.15, "flag_cos": 0.62, "flag_lex": 0.34, "min_title_words": 3,
            "s2_min_title_sim": 0.9, "s2_min_gain": 0.10, "use_s2": True}
_STOP = set("with from that this using based towards toward their into over under between through about within without when where which what your more most than then they them have been were also such".split())


def _words(t: str | None) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z0-9]{4,}", (t or "").lower()) if w not in _STOP}


def lexical_overlap(a: str | None, b: str | None) -> float:
    """|A ∩ B| / min(|A|, |B|) over 5-letter word stems; 1.0 when either side has no words."""
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 1.0


def title_in_abstract(title: str, abstract: str) -> float:
    tw = _words(title)
    return len(tw & _words(abstract)) / len(tw) if tw else 1.0


def is_latin(text: str | None) -> bool:
    t = text or ""
    return sum(ord(c) < 0x250 for c in t) / max(1, len(t)) > 0.9


def _norm(t: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (t or "").lower()))


_PLACEHOLDER = re.compile(r"^\s*(title pending|untitled|no title|title not available)", re.I)


def decide(rows: list[dict], P: dict) -> list[dict]:
    """Pure decision logic (tested offline). Each row: title, abstract, cos, reg_abstract, reg_cos (optional)."""
    dup_count = Counter()
    first = {}
    for i, r in enumerate(rows):
        key = _norm(r["abstract"])[:150]
        if len(key) >= 60:
            first[i] = key
            dup_count[(key, _norm(r["title"]))] += 0
    by_abs: dict[str, set] = {}
    for i, key in first.items():
        by_abs.setdefault(key, set()).add(_norm(rows[i]["title"]))
    out = []
    for i, r in enumerate(rows):
        d = {"action": "ok", "reason": None}
        t, a = r["title"], r["abstract"] or ""
        if len(a) < 40:
            out.append(d)
            continue
        lex = title_in_abstract(t, a)
        d["lex"] = round(lex, 3)
        reg = r.get("reg_abstract")
        if reg and lexical_overlap(a, reg) < P["registry_overlap"] and r.get("reg_cos", 0) >= r["cos"] + P["min_gain"]:
            d.update(action="replaced", reason="registry_mismatch", source=r.get("reg_source"), new_abstract=reg)
        elif _PLACEHOLDER.match(t):
            d.update(action="flagged", reason="placeholder_title")                         # the title is the problem here, not the abstract
        elif i in first and len(by_abs[first[i]]) >= 2 and r["cos"] < P["flag_cos"]:
            # a shared abstract is only evidence when it also fits the title badly: two versions of one paper (preprint / journal, different
            # titles) share a correct abstract, a publisher blurb does not fit any of its titles
            d.update(action="suspect", reason="duplicate_abstract")
        elif r["cos"] < P["low_cos"] and lex < P["low_lex"] and len(_words(t)) >= P["min_title_words"] and is_latin(a):
            d.update(action="suspect", reason="low_similarity")
        elif r["cos"] < P["flag_cos"] and lex < P["flag_lex"] and is_latin(a):
            d.update(action="flagged", reason="low_similarity_weak")
        out.append(d)
    return out


def run(cfg: Config) -> dict:
    P = {**DEFAULTS, **cfg.get("verify", {}).get("abstracts", {})}
    ver = load_verification(cfg)
    vrow = {r.paper_id: r for r in ver.itertuples()} if ver is not None else {}
    cache = cfg.paths.raw / "registry"
    reg = {"arxiv": Arxiv(cache).cache, "crossref": Crossref(cache).cache, "datacite": DataCite(cache).cache,
           "arxiv_title_search": Arxiv(cache).cache}
    works = {}
    for _, w in iter_candidates(cfg):
        works.setdefault(short_id(w["id"]), w)
    corpus = cfg.paths.processed / "corpus.parquet"
    if cfg["corpus"].get("fixed_from"):
        from ..config import PROJECT_ROOT
        corpus = PROJECT_ROOT / "data" / "processed" / cfg["corpus"]["fixed_from"] / "corpus.parquet"
    if corpus.exists():                                            # only the papers of this run (+ merged versions): the registry check is exact, the embedding is not free
        base = pd.read_parquet(corpus)
        keep = set(base["paper_id"]) | {a for alts in base["alt_ids"] for a in alts}
        if cfg["corpus"].get("fixed_from"):
            from .select_corpus import load_pool_works
            works = load_pool_works(cfg, keep)
        else:
            works = {k: v for k, v in works.items() if k in keep}
    ids, rows = [], []
    for pid, w in works.items():
        v = vrow.get(pid)
        title = clean_text(w.get("title") or w.get("display_name"))
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        if v is not None and v.status == "repaired":
            title, abstract = v.new_title, v.new_abstract
        r = reg.get(v.registry, {}).get(v.registry_id) if v is not None and v.registry else None
        ids.append(pid)
        rows.append({"title": title or "", "abstract": abstract or "", "reg_abstract": (r or {}).get("abstract"), "reg_source": v.registry if v is not None else None})
    ecfg = cfg.get("embedding", {})
    T = cached_embeddings(cfg, [r["title"] for r in rows], ecfg)
    A = cached_embeddings(cfg, [r["abstract"][:1500] for r in rows], ecfg)
    for r, t, a in zip(rows, T, A):
        r["cos"] = float(t @ a)
    need = [i for i, r in enumerate(rows) if r["reg_abstract"] and lexical_overlap(r["abstract"], r["reg_abstract"]) < P["registry_overlap"]]
    if need:
        R = cached_embeddings(cfg, [rows[i]["reg_abstract"][:1500] for i in need], ecfg)
        for i, e in zip(need, R):
            rows[i]["reg_cos"] = float(T[i] @ e)
    dec = decide(rows, P)

    suspects = [i for i, d in enumerate(dec) if d["action"] == "suspect"]
    if suspects and P["use_s2"]:
        s2 = S2Title(cache)
        got = s2.lookup(rows[i]["title"] for i in suspects)
        todo = [i for i in suspects if rows[i]["title"] in got and got[rows[i]["title"]].get("found") and got[rows[i]["title"]].get("abstract")]
        if todo:
            E = cached_embeddings(cfg, [got[rows[i]["title"]]["abstract"][:1500] for i in todo], ecfg)
            for i, e in zip(todo, E):
                rec = got[rows[i]["title"]]
                if title_similarity(rows[i]["title"], rec["title"]) >= P["s2_min_title_sim"] and float(T[i] @ e) >= rows[i]["cos"] + P["s2_min_gain"]:
                    dec[i].update(action="replaced", reason=f"{dec[i]['reason']}+s2_title_match", source="semanticscholar", new_abstract=rec["abstract"])
    for d in dec:
        if d["action"] == "suspect":
            d.update(action="dropped", reason=d["reason"] + "+no_replacement", new_abstract=None)

    df = pd.DataFrame([{"paper_id": pid, "title": r["title"], "cos_title_abstract": round(r["cos"], 4), "lex_title_in_abstract": d.get("lex"),
                        "action": d["action"], "reason": d["reason"], "source": d.get("source"), "new_abstract": d.get("new_abstract"),
                        "old_abstract": r["abstract"] if d["action"] in ("replaced", "dropped") else None}
                       for pid, r, d in zip(ids, rows, dec)])
    write_parquet(df, cfg.paths.processed / "abstract_check.parquet")
    changed = df[df.action.isin(["replaced", "dropped"])]
    report = {"papers": len(df), "actions": df.action.value_counts().to_dict(), "reasons": df.reason.fillna("-").value_counts().to_dict(), "params": P,
              "changed": changed[["paper_id", "title", "cos_title_abstract", "action", "reason", "source"]].to_dict("records")}
    write_json(cfg.paths.processed / "abstract_report.json", report)
    log.info("abstracts: %s", {k: report[k] for k in ("papers", "actions", "reasons")})
    return {"papers": len(df), "actions": report["actions"]}
