"""Semantic Scholar Graph API: reference lists for papers OpenAlex has none for.

POST /paper/batch accepts ARXIV:<id> / DOI:<doi> ids (100 per request here) and returns each
paper's references with their external ids, which we match against the corpus.
Anonymous access shares a global pool and returns 429 often; set S2_API_KEY in .env for a
dedicated rate limit. Cached in data/raw/semanticscholar/s2_references.jsonl.
"""
from __future__ import annotations

import os

from .registries import _Registry

URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS = "externalIds,referenceCount,references.externalIds"


class S2References(_Registry):
    name = "s2_references"
    batch_size = 100

    def __init__(self, cache_dir, **kw):
        kw.setdefault("max_retries", 12)
        super().__init__(cache_dir, **kw)
        self.min_interval_s = 1.0 if os.environ.get("S2_API_KEY") else 3.0
        if (key := os.environ.get("S2_API_KEY")):
            self.session.headers["x-api-key"] = key

    def _fetch_batch(self, keys):
        resp = self._get(URL, _json={"ids": keys}, fields=FIELDS)
        if resp is None or resp.status_code != 200:
            return None   # keeps failing: not cached (a not-found must not be cached by mistake)
        out = {}
        for key, item in zip(keys, resp.json()):  # results align with the request; null = unknown paper
            if not item:
                continue
            refs = []
            for r in item.get("references") or []:
                ext = r.get("externalIds") or {}
                if ext.get("ArXiv") or ext.get("DOI"):
                    refs.append({"arxiv": (ext.get("ArXiv") or "").lower() or None,
                                 "doi": (ext.get("DOI") or "").lower() or None})
            out[key] = {"key": key, "found": True, "reference_count": item.get("referenceCount") or 0,
                        "n_listed": len(item.get("references") or []), "refs": refs}
        return out
