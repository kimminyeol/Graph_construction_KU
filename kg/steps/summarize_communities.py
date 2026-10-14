"""Step `summarize`: LLM label + summary for every Community node (level 1 first, then level 0 with its children's labels).

  data/raw/llm_summary/<model>__<prompt>.jsonl      cache, one answer per distinct input (rerun = free)
  processed/<run>/communities.parquet               name / summary / keywords replaced (the TF-IDF terms are kept in `tfidf_keywords`)
  processed/<run>/community_summaries.json          full answers (+ subthemes, is_mixed), the evidence shown, and cost

Costs money (about $0.3 for 62 communities with gpt-4.1), so it only runs when named (`--steps summarize`) and stops at `summarize.max_usd`.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from .. import summarize as S
from ..config import Config
from ..io import write_json, write_parquet
from ..sources.llm import LLMClient
from .extract_entities import KNOWN_PRICES

log = logging.getLogger(__name__)
KNOWN_PRICES.setdefault("gpt-4.1", {"in": 2.00, "cached_in": 0.50, "out": 8.00})


def make_client(cfg: Config, scfg: dict) -> LLMClient:
    model = scfg.get("model", "gpt-4.1")
    price = scfg.get("price") or KNOWN_PRICES.get(model)
    path = cfg.paths.raw / "llm_summary" / f"{model}__{scfg.get('prompt_version', S.PROMPT_VERSION)}.jsonl"
    return LLMClient(model, path, params=scfg.get("params", {"temperature": 0, "max_tokens": 900}), price=price,
                     max_usd=scfg.get("max_usd"), workers=scfg.get("workers", 8))


def summarize_level(client: LLMClient, inputs: dict, ids: list[str], version: str, extra: dict | None = None) -> dict[str, dict]:
    system = S.PROMPTS[version]
    items, key_of = [], {}
    for cid in ids:
        payload = {**inputs[cid]["payload"], **((extra or {}).get(cid, {}))}
        k = S.input_key(payload, version)
        key_of[cid] = k
        items.append((k, system, json.dumps(payload, ensure_ascii=False, indent=1)))
    client.run_many(items, S.SCHEMA)
    return {cid: client.cache[k]["content"] for cid, k in key_of.items() if k in client.cache}


def run(cfg: Config) -> dict:
    scfg = dict(cfg.get("summarize", {}))
    version = scfg.get("prompt_version", S.PROMPT_VERSION)
    out = cfg.paths.processed
    t = S.load_graph_tables(out)
    inputs = S.build_inputs(t, scfg.get("n_central", 8), scfg.get("n_random", 4), use_entities=scfg.get("entities", True))
    client = make_client(cfg, scfg)
    l1 = [c for c, v in inputs.items() if v["level"] == 1]
    l0 = [c for c, v in inputs.items() if v["level"] == 0]
    ans = summarize_level(client, inputs, l1, version)
    extra = {c: {"sub_communities": [f"{ans[ch]['name']} ({inputs[c]['child_sizes'][ch]} papers)" for ch in inputs[c]["children"] if ch in ans]} for c in l0}
    ans.update(summarize_level(client, inputs, l0, version, extra))
    comm = t["comm"].copy()
    if "tfidf_keywords" not in comm:
        comm["tfidf_keywords"] = comm["keywords"]
        comm["tfidf_name"] = comm["name"]
    cid = comm.node_id.str.split(":", n=1).str[1]
    comm["name"] = [ans[c]["name"] if c in ans else n for c, n in zip(cid, comm["name"])]
    comm["summary"] = [ans[c]["summary"] if c in ans else None for c in cid]
    comm["keywords"] = [list(ans[c]["keywords"]) if c in ans else list(k) for c, k in zip(cid, comm["tfidf_keywords"])]
    comm["subthemes"] = [list(ans[c]["subthemes"]) if c in ans else [] for c in cid]
    comm["is_mixed"] = [bool(ans[c]["is_mixed"]) if c in ans else None for c in cid]
    write_parquet(comm, out / "communities.parquet")
    write_json(out / "community_summaries.json", {"model": client.model, "prompt_version": version, "answers": ans,
               "evidence": {c: {"payload": v["payload"], "shown": v["shown"]} for c, v in inputs.items()},
               "tokens": client.tokens, "spent_usd": round(client.spent, 4), "cache_total_cost_usd": client.cached_cost()})
    log.info("summarize: %d/%d communities answered, spent $%.3f", len(ans), len(inputs), client.spent)
    return {"answered": len(ans), "communities": len(inputs), "spent_usd": round(client.spent, 4)}
