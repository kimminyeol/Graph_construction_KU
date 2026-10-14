"""OpenAlex REST client with retries, cursor paging and a spend guard.

OpenAlex bills per call (list/filter ~$0.0001, search ~$0.001) against a daily
allowance ($0.10/day anonymous). Remaining budget is read from the
`x-ratelimit-remaining-usd` response header; we stop *before* hitting zero so
an interrupted collection can resume the next day from the saved cursor.
Set OPENALEX_API_KEY (in .env) for a larger allowance and no anonymous
search throttling.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    spent_usd: float = 0.0
    remaining_usd: float | None = None
    retries: int = 0
    history: list[float] = field(default_factory=list, repr=False)


class OpenAlexClient:
    def __init__(self, api_key: str | None = None, mailto: str | None = None,
                 min_interval_s: float = 0.15, max_retries: int = 8, max_throttle_waits: int = 40,
                 timeout_s: float = 90,
                 min_remaining_usd: float = 0.005):
        self.api_key = api_key or os.environ.get("OPENALEX_API_KEY") or None
        self.mailto = mailto or os.environ.get("OPENALEX_MAILTO") or None
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.max_throttle_waits = max_throttle_waits
        self.timeout_s = timeout_s
        self.min_remaining_usd = min_remaining_usd
        self.usage = Usage()
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "research-kg-builder/0.1"

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _record(self, resp: requests.Response) -> None:
        h = resp.headers
        self.usage.calls += 1
        if (cost := h.get("x-ratelimit-cost-usd")) is not None:
            self.usage.spent_usd += float(cost)
        if (rem := h.get("x-ratelimit-remaining-usd")) is not None:
            self.usage.remaining_usd = float(rem)

    @staticmethod
    def _retry_after(resp: requests.Response, attempt: int) -> float:
        if (ra := resp.headers.get("Retry-After")):
            try:
                return float(ra) + 1
            except ValueError:
                pass
        m = re.search(r"retry in (\d+)s", resp.text or "")
        if m:
            return float(m.group(1)) + 1
        return min(2 ** attempt, 120)

    def get(self, path: str, params: dict[str, Any]) -> dict:
        if self.usage.remaining_usd is not None and self.usage.remaining_usd < self.min_remaining_usd:
            raise BudgetExhausted(f"OpenAlex budget nearly exhausted (${self.usage.remaining_usd:.4f} left)")
        params = {k: v for k, v in params.items() if v is not None}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto
        url = f"{BASE_URL}/{path.lstrip('/')}"
        errors = throttles = 0  # transient failures vs. rate-limit waits have separate budgets
        while True:
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout_s)
            except requests.RequestException as e:
                errors += 1
                if errors > self.max_retries:
                    raise
                wait = min(2 ** errors, 60)
                log.warning("request error %s; retry %d/%d in %.0fs", e, errors, self.max_retries, wait)
                self.usage.retries += 1
                time.sleep(wait)
                continue
            self._record(resp)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                if self.usage.remaining_usd is not None and self.usage.remaining_usd < self.min_remaining_usd:
                    raise BudgetExhausted(f"daily budget exhausted: {resp.text[:200]}")
                # anonymous search is throttled for tens of seconds at a time under load: wait it out
                throttles += 1
                if throttles > self.max_throttle_waits:
                    raise BudgetExhausted(f"still rate-limited after {throttles - 1} waits: {resp.text[:200]}")
                wait = self._retry_after(resp, throttles)
                log.warning("HTTP 429 (%s); wait %d/%d, %.0fs", resp.text[:100].replace("\n", " "),
                            throttles, self.max_throttle_waits, wait)
            elif resp.status_code in (500, 502, 503, 504) and errors < self.max_retries:
                errors += 1
                wait = self._retry_after(resp, errors)
                log.warning("HTTP %s; retry %d/%d in %.0fs", resp.status_code, errors, self.max_retries, wait)
            else:
                raise RuntimeError(f"OpenAlex {resp.status_code} for {resp.url}: {resp.text[:500]}")
            self.usage.retries += 1
            time.sleep(wait)

    def iter_pages(self, path: str, params: dict[str, Any], cursor: str = "*",
                   per_page: int = 200) -> Iterator[tuple[list[dict], str | None, int]]:
        """Yield (results, next_cursor, total_count) until the cursor runs out."""
        while cursor:
            data = self.get(path, {**params, "per_page": per_page, "cursor": cursor})
            cursor = data["meta"].get("next_cursor")
            yield data["results"], cursor, data["meta"].get("count", 0)
            if not data["results"]:
                return


def build_filter(filters: dict[str, Any]) -> str:
    """{'publication_year': 2024, 'type': ['article', 'preprint']} -> 'publication_year:2024,type:article|preprint'."""
    parts = []
    for k, v in filters.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = str(v).lower()
        elif isinstance(v, (list, tuple)):
            v = "|".join(str(x) for x in v)
        v = str(v)
        if "," in v:
            raise ValueError(f"filter value for {k!r} may not contain commas: {v!r}")
        parts.append(f"{k}:{v}")
    return ",".join(parts)


def short_id(url_or_id: str | None) -> str | None:
    """'https://openalex.org/W123' -> 'W123'; 'https://openalex.org/subfields/1702' -> 'subfields/1702'."""
    if not url_or_id:
        return None
    return url_or_id.replace("https://openalex.org/", "")


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted:
        return None
    pos = {i: w for w, idxs in inverted.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos)) or None
