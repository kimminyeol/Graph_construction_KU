"""OpenAI chat-completions client for structured extraction.

  * every successful response is appended to a JSONL cache at once, so a stopped run resumes where it ended
    and later runs (other configs, other graph variants) reuse the answers for free
  * 429 / 5xx are retried with backoff (Retry-After honoured); a spend guard stops the run at `max_usd`
  * model-specific request parameters (temperature, max_tokens, reasoning_effort, ...) come from the config
    verbatim, so switching models never needs code changes

The API key is read from the environment (`OPENAI_API_KEY`, loaded from .env) and is never logged.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from ..io import iter_jsonl

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, model: str, cache_path: Path, *, params: dict | None = None, price: dict | None = None,
                 max_usd: float | None = None, workers: int = 8, timeout: float = 180.0, max_retries: int = 6,
                 api_key_env: str = "OPENAI_API_KEY", base_url: str = "https://api.openai.com/v1"):
        self.model, self.cache_path = model, Path(cache_path)
        self.params = dict(params or {})
        self.price = price or {}            # USD per 1M tokens: in, cached_in, out (unknown -> cost not estimated)
        self.max_usd, self.workers, self.timeout, self.max_retries = max_usd, workers, timeout, max_retries
        self.url = base_url.rstrip("/") + "/chat/completions"
        self._key = os.environ.get(api_key_env)      # only needed when a request is not in the cache
        self._key_env = api_key_env
        self._lock = threading.Lock()
        self.cache: dict[str, dict] = {r["key"]: r for r in iter_jsonl(self.cache_path)}
        self.spent = 0.0
        self.tokens = {"in": 0, "cached_in": 0, "out": 0, "reasoning": 0}
        self.stats = {"calls": 0, "cached": 0, "failed": 0, "stopped_by_budget": False}

    # --- cost -------------------------------------------------------------------------------------------------------
    def cost_of(self, usage: dict) -> float | None:
        if "in" not in self.price or "out" not in self.price:
            return None
        cached = usage.get("cached_in", 0)
        return ((usage.get("in", 0) - cached) * self.price["in"] + cached * self.price.get("cached_in", self.price["in"])
                + usage.get("out", 0) * self.price["out"]) / 1e6

    def cached_cost(self) -> float | None:
        costs = [self.cost_of(r.get("usage", {})) for r in self.cache.values()]
        return None if any(c is None for c in costs) else sum(costs)

    # --- one request ------------------------------------------------------------------------------------------------
    def _request(self, system: str, user: str, schema: dict) -> dict | None:
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "response_format": {"type": "json_schema", "json_schema": {"name": "extraction", "strict": True, "schema": schema}},
                   **self.params}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                log.warning("request error (%s), retry %d", type(e).__name__, attempt + 1)
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After") or min(60, 2 ** (attempt + 1)))
                log.warning("HTTP %d, waiting %.0fs", r.status_code, wait)
                time.sleep(wait)
                continue
            # 4xx other than 429: a bad request will not get better; show the API's reason (never the key)
            log.error("HTTP %d: %s", r.status_code, r.text[:300].replace(self._key, "***"))
            return None
        return None

    def _one(self, key: str, system: str, user: str, schema: dict) -> dict | None:
        if self.max_usd is not None and self.spent >= self.max_usd:
            self.stats["stopped_by_budget"] = True
            return None
        resp = self._request(system, user, schema)
        if resp is None:
            return None
        try:
            choice = resp["choices"][0]
            content = json.loads(choice["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            log.warning("unparseable response for %s (finish=%s)", key, (resp.get("choices") or [{}])[0].get("finish_reason"))
            return None
        u = resp.get("usage") or {}
        usage = {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0),
                 "cached_in": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                 "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)}
        rec = {"key": key, "model": resp.get("model", self.model), "content": content, "usage": usage,
               "finish": choice.get("finish_reason"), "ts": int(time.time())}
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.cache[key] = rec
            for k in self.tokens:
                self.tokens[k] += usage[k]
            self.stats["calls"] += 1
            self.spent += self.cost_of(usage) or 0.0
        return rec

    # --- many requests ----------------------------------------------------------------------------------------------
    def run_many(self, items: list[tuple[str, str, str]], schema: dict, progress_every: int = 200) -> dict:
        """items = (cache_key, system, user). Already cached keys are skipped."""
        todo = [it for it in items if it[0] not in self.cache]
        self.stats["cached"] = len(items) - len(todo)
        if todo and not self._key:
            raise RuntimeError(f"{len(todo)} requests are not cached and {self._key_env} is not set (put it in .env)")
        log.info("llm[%s]: %d requests (%d already cached), %d workers", self.model, len(todo), self.stats["cached"], self.workers)
        t0, done = time.time(), 0
        with ThreadPoolExecutor(self.workers) as ex:
            futs = {ex.submit(self._one, k, s, u, schema): k for k, s, u in todo}
            for fut in as_completed(futs):
                done += 1
                if fut.result() is None:
                    self.stats["failed"] += 1
                if done % progress_every == 0:
                    rate = done / (time.time() - t0)
                    log.info("llm: %d/%d  %.1f req/s  spent=%s  failed=%d", done, len(todo), rate,
                             f"${self.spent:.2f}" if self.price else f"{self.tokens['out']} out tokens", self.stats["failed"])
        return {**self.stats, "tokens": dict(self.tokens), "spent_usd": round(self.spent, 4) if self.price else None,
                "seconds": round(time.time() - t0, 1)}
