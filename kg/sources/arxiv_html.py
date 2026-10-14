"""arXiv HTML bibliographies (rule-based, no LLM).

arxiv.org/html/<id> (and ar5iv.labs.arxiv.org/html/<id> for older papers) render the reference list as
<li class="ltx_bibitem"> elements. We fetch the page, cut the bibliography into entries (clean text +
the arXiv ids / DOIs found in the entry's links) and cache the result in
data/raw/arxiv_html/bibliography.jsonl, so every run only fetches papers it has not seen.
"""
from __future__ import annotations

import html as _html
import logging
import re
import time
from pathlib import Path

import requests

from ..io import append_jsonl, iter_jsonl

log = logging.getLogger(__name__)

_BIBITEM = re.compile(r'<li[^>]*class="[^"]*ltx_bibitem[^"]*"[^>]*>(.*?)</li>', re.S)
_HREF = re.compile(r'href="([^"]+)"')
_ARXIV_ID = re.compile(r"(?:arxiv[:\s./]*(?:abs/|pdf/)?|arxiv\.org/(?:abs|pdf)/)((?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?", re.I)
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def find_ids(text: str) -> tuple[list[str], list[str]]:
    """arXiv ids and DOIs mentioned in a piece of text or URL list."""
    arx = [m.group(1).lower() for m in _ARXIV_ID.finditer(text)]
    dois = [d.rstrip(".,;)").lower() for d in _DOI.findall(text)]
    return list(dict.fromkeys(arx)), list(dict.fromkeys(dois))


def parse_bibliography(page: str) -> list[dict]:
    entries = []
    for i, raw in enumerate(_BIBITEM.findall(page)):
        links = _HREF.findall(raw)
        text = clean(raw)
        arx, dois = find_ids(text + " " + " ".join(links))
        entries.append({"idx": i, "text": text[:1500], "arxiv_ids": arx, "dois": dois})
    return entries


class ArxivBibliographies:
    def __init__(self, cache_dir: Path, min_interval_s: float = 0.6, max_retries: int = 4):
        self.path = cache_dir / "bibliography.jsonl"
        self.cache = {r["arxiv_id"]: r for r in iter_jsonl(self.path)}
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "research-kg-builder/0.1 (reference parsing; polite rate)"
        self._last = 0.0
        self.fetched = 0

    def _get(self, url: str) -> requests.Response | None:
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(url, timeout=90)
            except requests.RequestException as e:
                log.warning("arxiv html error %s; retry %d", e, attempt + 1)
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(float(r.headers.get("Retry-After", 0) or 0) or min(2 ** attempt * 5, 90))
                continue
            return r
        return None

    def get(self, arxiv_id: str) -> dict:
        """{'arxiv_id', 'found', 'url', 'entries': [...]}; cached, 'not found' is cached too."""
        if arxiv_id in self.cache:
            return self.cache[arxiv_id]
        rec = {"arxiv_id": arxiv_id, "found": False, "url": None, "entries": []}
        for url in (f"https://arxiv.org/html/{arxiv_id}", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"):
            r = self._get(url)
            if r is not None and r.status_code == 200:
                entries = parse_bibliography(r.text)
                if entries:
                    rec.update(found=True, url=url, entries=entries)
                    break
        self.fetched += 1
        self.cache[arxiv_id] = rec
        append_jsonl(self.path, [rec])
        return rec
