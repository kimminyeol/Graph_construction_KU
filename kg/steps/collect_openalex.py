"""Step `collect`: fetch candidate works from OpenAlex into the shared raw store.

The corpus spec is expanded into *units* (source item x year). Each unit has a
quota and its own append-only JSONL file + saved cursor, so:
  * an interrupted run resumes where it stopped,
  * raising `max_papers` later only fetches the extra pages,
  * changing a unit's query (filters/sort/fields) re-fetches just that unit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import Config
from ..io import append_jsonl, read_json, write_json
from ..sources.openalex import BudgetExhausted, OpenAlexClient, build_filter

log = logging.getLogger(__name__)

WORK_FIELDS = [
    "id", "doi", "ids", "title", "display_name", "publication_year", "publication_date", "type",
    "language", "primary_location", "open_access", "authorships", "primary_topic", "topics",
    "keywords", "cited_by_count", "fwci", "referenced_works_count", "referenced_works",
    "abstract_inverted_index", "is_retracted", "is_paratext",
]


@dataclass(frozen=True)
class Unit:
    key: str
    source: str
    kind: str          # topic | search
    item: str
    year: int | None
    quota: int         # papers this unit should contribute to the corpus
    filters: tuple     # frozen (k, v) pairs


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def plan_units(cfg: Config) -> list[Unit]:
    c = cfg["corpus"]
    y0, y1 = c["years"]
    years: list[int | None] = list(range(y0, y1 + 1)) if c.get("stratify_by_year", True) else [None]
    total_share = sum(s["share"] for s in c["sources"])
    units = []
    for src in c["sources"]:
        items = list(src["items"])  # dict (id -> label) or list
        per_unit = math.ceil(c["max_papers"] * src["share"] / total_share / len(items) / len(years))
        for item in items:
            for year in years:
                f = dict(c.get("base_filters", {}))
                f.update(src.get("extra_filters", {}))
                f["publication_year"] = year if year is not None else f"{y0}-{y1}"
                if src["kind"] == "topic":
                    f["primary_topic.id"] = item
                elif src["kind"] == "search":
                    f["title_and_abstract.search"] = item
                else:
                    raise ValueError(f"unknown source kind {src['kind']!r}")
                key = f"{src['name']}__{_slug(item)}__{year or 'all'}"
                units.append(Unit(key, src["name"], src["kind"], item, year, per_unit,
                                  tuple(sorted((k, json.dumps(v)) for k, v in f.items()))))
    return units


def _unit_query(unit: Unit, cfg: Config) -> dict:
    filters = {k: json.loads(v) for k, v in unit.filters}
    return {"filter": build_filter(filters), "sort": cfg["corpus"].get("sort"), "select": ",".join(WORK_FIELDS)}


def _signature(query: dict) -> str:
    return hashlib.sha1(json.dumps(query, sort_keys=True).encode()).hexdigest()[:12]


def run(cfg: Config) -> dict:
    raw_dir = cfg.paths.openalex_raw
    works_dir = raw_dir / "works"
    state_path = raw_dir / "collect_state.json"
    state: dict = read_json(state_path, {})
    oa = cfg.get("openalex", {})
    client = OpenAlexClient(min_interval_s=oa.get("min_interval_s", 0.15),
                            min_remaining_usd=oa.get("min_remaining_usd", 0.005))
    units = plan_units(cfg)
    oversample = cfg["corpus"].get("oversample", 1.5)
    log.info("collect: %d units, api_key=%s", len(units), "yes" if client.api_key else "no (anonymous)")

    fetched_total, stopped_early = 0, None
    for i, unit in enumerate(units, 1):
        query = _unit_query(unit, cfg)
        sig = _signature(query)
        path = works_dir / f"{unit.key}.jsonl"
        st = state.get(unit.key)
        if st is None or st["signature"] != sig:
            if path.exists():
                log.info("[%s] query changed -> refetch", unit.key)
                path.unlink()
            st = {"signature": sig, "cursor": "*", "n_fetched": 0, "total_count": None,
                  "exhausted": False, "query": query}
        target = math.ceil(unit.quota * oversample)
        if st["exhausted"] or st["n_fetched"] >= target:
            state[unit.key] = st
            continue
        try:
            per_page = max(25, min(200, target - st["n_fetched"]))
            for results, next_cursor, count in client.iter_pages("works", query, st["cursor"], per_page):
                for r in results:
                    r["_unit"] = unit.key
                append_jsonl(path, results)
                st.update(n_fetched=st["n_fetched"] + len(results), cursor=next_cursor, total_count=count,
                          exhausted=not next_cursor or not results,
                          updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                state[unit.key] = st
                write_json(state_path, state)  # persist after every page
                fetched_total += len(results)
                if st["exhausted"] or st["n_fetched"] >= target:
                    break
        except BudgetExhausted as e:
            stopped_early = str(e)
            log.warning("stopping: %s. Re-run `collect` later (or add OPENALEX_API_KEY) to resume.", e)
            break
        log.info("[%d/%d] %-60s %4d/%-4d (matches: %s)", i, len(units), unit.key,
                 st["n_fetched"], target, st["total_count"])
    write_json(state_path, state)

    report = {"units": len(units), "fetched_this_run": fetched_total,
              "api_calls": client.usage.calls, "spent_usd": round(client.usage.spent_usd, 5),
              "remaining_usd": client.usage.remaining_usd, "retries": client.usage.retries,
              "complete": stopped_early is None, "stopped_reason": stopped_early}
    write_json(cfg.paths.processed / "collect_report.json", report)
    log.info("collect: %s", report)
    return report
