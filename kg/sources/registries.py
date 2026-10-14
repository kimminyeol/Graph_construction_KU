"""Authoritative metadata registries used to verify OpenAlex records.

OpenAlex sometimes has a work's title/abstract (or DOI) overwritten by an
unrelated record while its citations still belong to the original paper,
e.g. W4226278401 is InstructGPT (arXiv 2203.02155, Ouyang et al.) but carries
the title "Discernment and Social Learning as a Companion Training Layer".
We check records against the registry that owns their identifier:
arXiv (arXiv ids) -> Crossref (most DOIs) -> DataCite (Dagstuhl, Zenodo, ...).

Every lookup (including "not found") is cached append-only under
data/raw/registry/<name>.jsonl, so scaling up only queries new ids.
"""
from __future__ import annotations

import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import requests

from ..io import append_jsonl, iter_jsonl

log = logging.getLogger(__name__)

_ATOM = "{http://www.w3.org/2005/Atom}"
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _clean(s: str | None) -> str | None:
    if not s:
        return None
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", s))).strip() or None


class _Registry:
    name = "registry"
    min_interval_s = 0.2
    batch_size = 50

    def __init__(self, cache_dir: Path, max_retries: int = 5):
        self.cache_path = cache_dir / f"{self.name}.jsonl"
        self.cache = {r["key"]: r for r in iter_jsonl(self.cache_path)}
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "research-kg-builder/0.1"
        self._last = 0.0
        self.calls = 0

    def _get(self, url: str, _json: dict | None = None, **params) -> requests.Response | None:
        """GET (or POST when `_json` is given) with throttling and retries; None if it keeps failing."""
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                if _json is not None:
                    resp = self.session.post(url, params=params or None, json=_json, timeout=120)
                else:
                    resp = self.session.get(url, params=params or None, timeout=60)
                self.calls += 1
            except requests.RequestException as e:
                log.warning("%s: %s; retry %d", self.name, e, attempt + 1)
                time.sleep(min(2 ** attempt * 3, 60))
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                ra = resp.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit() else min(2 ** attempt * 3, 60))
                continue
            return resp
        log.warning("%s: giving up on %s", self.name, url)
        return None

    def _fetch_batch(self, keys: list[str]) -> dict[str, dict]:
        raise NotImplementedError

    def lookup(self, keys: Iterable[str]) -> dict[str, dict]:
        """key -> record (``found`` False if the registry does not know it)."""
        keys = list(dict.fromkeys(k for k in keys if k))
        todo = [k for k in keys if k not in self.cache]
        if todo:
            log.info("%s: %d cached, %d to fetch", self.name, len(keys) - len(todo), len(todo))
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            got = self._fetch_batch(batch)
            if got is None:  # transient failure: do not cache, retry on the next run
                continue
            rows = [got.get(k) or {"key": k, "found": False} for k in batch]
            append_jsonl(self.cache_path, rows)
            self.cache.update({r["key"]: r for r in rows})
        return {k: self.cache[k] for k in keys if k in self.cache}


class Arxiv(_Registry):
    name = "arxiv"
    min_interval_s = 3.0   # arXiv API terms: at most one request every 3 seconds
    batch_size = 100

    def _fetch_batch(self, keys):
        resp = self._get("https://export.arxiv.org/api/query", id_list=",".join(keys), max_results=len(keys))
        if resp is None:
            return None
        if resp.status_code != 200:
            if len(keys) == 1:
                return {}  # malformed id -> not found
            # one bad id fails the whole batch: bisect
            mid = len(keys) // 2
            a, b = self._fetch_batch(keys[:mid]), self._fetch_batch(keys[mid:])
            return None if a is None or b is None else {**a, **b}
        out = {}
        for e in ET.fromstring(resp.content).findall(f"{_ATOM}entry"):
            eid = (e.findtext(f"{_ATOM}id") or "").split("/abs/")[-1]
            key = re.sub(r"v\d+$", "", eid)
            title = _clean(e.findtext(f"{_ATOM}title"))
            if not key or not title or title == "Error":
                continue
            out[key] = self._entry(e, key, title)
        return out

    @staticmethod
    def _entry(e, key: str, title: str) -> dict:
        return {"key": key, "found": True, "title": title,
                "abstract": _clean(e.findtext(f"{_ATOM}summary")),
                "year": int((e.findtext(f"{_ATOM}published") or "0")[:4]) or None,
                "authors": [_clean(a.findtext(f"{_ATOM}name")) for a in e.findall(f"{_ATOM}author")]}

    def search_title(self, title: str) -> list[dict]:
        """Top arXiv hits for a title (cached). Used to tell a corrupt DOI from a corrupt title."""
        # lower-case words of 4+ chars: avoids AND/OR/NOT being read as query operators
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", title) if len(w) >= 4][:12]
        if not words:
            return []
        key = "title:" + " ".join(words)
        if key not in self.cache:
            resp = self._get("https://export.arxiv.org/api/query",
                             search_query=" AND ".join(f"ti:{w}" for w in words), max_results=5)
            if resp is None or resp.status_code != 200:
                return []
            hits = []
            for e in ET.fromstring(resp.content).findall(f"{_ATOM}entry"):
                t = _clean(e.findtext(f"{_ATOM}title"))
                eid = re.sub(r"v\d+$", "", (e.findtext(f"{_ATOM}id") or "").split("/abs/")[-1])
                if t and eid:
                    hits.append(self._entry(e, eid, t))
            row = {"key": key, "found": bool(hits), "hits": hits}
            append_jsonl(self.cache_path, [row])
            self.cache[key] = row
        return self.cache[key].get("hits", [])


class Crossref(_Registry):
    name = "crossref"
    batch_size = 40

    def _fetch_batch(self, keys):
        resp = self._get("https://api.crossref.org/works", rows=len(keys) * 2,
                         filter=",".join(f"doi:{k}" for k in keys),
                         select="DOI,title,abstract,issued,author")
        if resp is None:
            return None
        if resp.status_code != 200:
            if len(keys) == 1:
                return {}
            mid = len(keys) // 2
            a, b = self._fetch_batch(keys[:mid]), self._fetch_batch(keys[mid:])
            return None if a is None or b is None else {**a, **b}
        out = {}
        for it in resp.json()["message"]["items"]:
            key = it["DOI"].lower()
            parts = (it.get("issued") or {}).get("date-parts") or [[None]]
            out[key] = {"key": key, "found": True, "title": _clean((it.get("title") or [None])[0]),
                        "abstract": _clean(it.get("abstract")), "year": parts[0][0],
                        "authors": [" ".join(filter(None, (a.get("given"), a.get("family") or a.get("name"))))
                                    for a in it.get("author") or []]}
        return out


class DataCite(_Registry):
    name = "datacite"
    batch_size = 1

    def _fetch_batch(self, keys):
        resp = self._get(f"https://api.datacite.org/dois/{keys[0]}")
        if resp is None:
            return None
        if resp.status_code != 200:
            return {}
        a = resp.json()["data"]["attributes"]
        abstract = next((d.get("description") for d in a.get("descriptions") or []
                         if d.get("descriptionType") == "Abstract"), None)
        return {keys[0]: {"key": keys[0], "found": True,
                          "title": _clean((a.get("titles") or [{}])[0].get("title")),
                          "abstract": _clean(abstract), "year": a.get("publicationYear"),
                          "authors": [c.get("name") for c in a.get("creators") or []]}}


class S2Title(_Registry):
    """Semantic Scholar best title match -> {title, abstract, year}. key = the title as asked. Anonymous access shares a pool and is
    rate-limited (429 is retried with backoff by `_get`); set S2_API_KEY in .env for a dedicated limit."""
    name = "s2_title"
    batch_size = 1
    min_interval_s = 3.0

    def __init__(self, cache_dir, **kw):
        super().__init__(cache_dir, **kw)
        import os
        if (key := os.environ.get("S2_API_KEY")):
            self.session.headers["x-api-key"] = key
            self.min_interval_s = 1.0

    def _fetch_batch(self, keys):
        r = self._get("https://api.semanticscholar.org/graph/v1/paper/search/match", query=keys[0], fields="title,abstract,year")
        if r is None:
            return None
        if r.status_code == 404:                       # "no match"
            return {keys[0]: {"key": keys[0], "found": False}}
        if r.status_code != 200:
            return None
        d = (r.json().get("data") or [None])[0]
        if not d:
            return {keys[0]: {"key": keys[0], "found": False}}
        return {keys[0]: {"key": keys[0], "found": True, "title": d.get("title"), "abstract": _clean(d.get("abstract")), "year": d.get("year")}}
