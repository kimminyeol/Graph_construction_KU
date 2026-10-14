"""Step `extract`: title + abstract -> entities / results with an LLM (OpenAI API), cached per (model, prompt version).

  data/raw/llm_extract/<model>__<prompt_version>[__<tag>].jsonl   shared by every run (one JSON object per paper)
  processed/<run>/extract_report.json                              counts, tokens, cost

It stops at `extract.max_usd`. Rerunning is free: cached papers are skipped. Abstract only (no full text) by design: cheapest,
and every paper has one.
"""
from __future__ import annotations

import logging

import pandas as pd

from .. import extraction
from ..config import Config
from ..io import read_parquet, write_json
from ..sources.llm import LLMClient

log = logging.getLogger(__name__)

# USD per 1M tokens (input, cached input, output). Recalled from public price lists: verify before relying on the budget guard.
KNOWN_PRICES = {
    "gpt-4.1": {"in": 2.00, "cached_in": 0.50, "out": 8.00},
    "gpt-4.1-mini": {"in": 0.40, "cached_in": 0.10, "out": 1.60},
    "gpt-4.1-nano": {"in": 0.10, "cached_in": 0.025, "out": 0.40},
    "gpt-4o-mini": {"in": 0.15, "cached_in": 0.075, "out": 0.60},
    "gpt-5-mini": {"in": 0.25, "cached_in": 0.025, "out": 2.00},
    "gpt-5-nano": {"in": 0.05, "cached_in": 0.005, "out": 0.40},
}


def cache_name(ecfg: dict) -> str:
    return "__".join(x for x in (ecfg["model"], ecfg.get("prompt_version", extraction.PROMPT_VERSION), ecfg.get("tag")) if x)


def cache_file(cfg: Config, ecfg: dict):
    return cfg.paths.raw / "llm_extract" / f"{cache_name(ecfg)}.jsonl"


def make_client(cfg: Config, ecfg: dict) -> LLMClient:
    price = ecfg.get("price") or KNOWN_PRICES.get(ecfg["model"])
    if ecfg.get("max_usd") is not None and not price:
        raise ValueError(f"extract.max_usd is set but no price is known for {ecfg['model']}: add extract.price {{in, cached_in, out}} (USD per 1M tokens)")
    return LLMClient(ecfg["model"], cache_file(cfg, ecfg), params=ecfg.get("params"), price=price, max_usd=ecfg.get("max_usd"),
                     workers=ecfg.get("workers", 16))


def select_papers(cfg: Config, ecfg: dict) -> pd.DataFrame:
    papers = read_parquet(cfg.paths.processed / "papers.parquet")[["node_id", "source_id", "name", "abstract"]]
    papers = papers[papers["name"].notna() & papers["abstract"].fillna("").str.len().ge(ecfg.get("min_abstract_chars", 200))]
    if (only := ecfg.get("only_file")):
        sel = pd.read_csv(only, dtype=str)
        if (split := ecfg.get("only_split")) and "split" in sel:
            sel = sel[sel["split"] == split]
        papers = papers[papers["source_id"].isin(sel["paper"])]
    if (limit := ecfg.get("limit")):
        papers = papers.head(int(limit))
    return papers


def run(cfg: Config) -> dict:
    ecfg = dict(cfg.get("extract", {}))
    papers = select_papers(cfg, ecfg)
    system = extraction.system_prompt(ecfg.get("prompt_version", extraction.PROMPT_VERSION))
    client = make_client(cfg, ecfg)
    items, seen = [], set()
    for r in papers.itertuples():
        k = extraction.cache_key(r.name, r.abstract)
        if k not in seen:
            seen.add(k)
            items.append((k, system, extraction.user_message(r.name, r.abstract)))
    res = client.run_many(items, extraction.SCHEMA)
    done = sum(k in client.cache for k, _, _ in items)
    report = {"model": ecfg["model"], "prompt_version": ecfg.get("prompt_version", extraction.PROMPT_VERSION), "papers": len(papers),
              "requests": len(items), "answered": done, "cache_file": str(client.cache_path), **res,
              "cache_total_cost_usd": client.cached_cost()}
    write_json(cfg.paths.processed / "extract_report.json", report)
    log.info("extract: %s", {k: v for k, v in report.items() if k != "cache_file"})
    if res["failed"] or done < len(items):
        log.warning("extract: %d of %d papers have no answer (budget stop or API errors); rerun to resume", len(items) - done, len(items))
    return report
