"""Step `build`: OpenAlex metadata -> Base Research KG tables (no LLM).

Nodes: Paper, Author, Institution, Venue, Topic (+ subfield/field/domain levels)
Edges: AUTHORED, AFFILIATED_WITH, PUBLISHED_AT, HAS_TOPIC, SUBTOPIC_OF, CITES
Output: processed/<run>/{papers,authors,institutions,venues,topics,edges}.parquet
"""
from __future__ import annotations

import html
import json
import logging
import re
from collections import defaultdict
from typing import Iterable

import pandas as pd

from .. import schema
from ..config import Config
from ..io import iter_jsonl, read_parquet, write_json, write_parquet
from ..sources.openalex import reconstruct_abstract, short_id
from .collect_openalex import plan_units
from .select_corpus import load_abstract_fixes, load_verification

log = logging.getLogger(__name__)
SOURCE = "openalex"

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(s: str | None) -> str | None:
    if not s:
        return None
    return _WS.sub(" ", html.unescape(_TAG.sub("", s))).strip() or None


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def author_node_id(author: dict) -> tuple[str | None, str]:
    """OpenAlex author id > ORCID > normalised name (flagged as unresolved)."""
    if author.get("id"):
        return schema.make_id("Author", short_id(author["id"])), "openalex"
    if author.get("orcid"):
        return schema.make_id("Author", "orcid-" + author["orcid"].rsplit("/", 1)[-1]), "orcid"
    name = author.get("display_name")
    if name and _slug(name):
        return schema.make_id("Author", "name-" + _slug(name)), "name"
    return None, "none"


def author_canonical_map(works: list[dict], min_overlap: float = 0.5) -> dict[str, str]:
    """Author node id -> canonical node id, merging nodes that share an ORCID.

    OpenAlex gives one person several author ids, and our ORCID fallback id (`author:orcid-...`) can sit next to the
    OpenAlex id of the same person. But an ORCID is sometimes attached to the wrong author in a raw authorship
    (e.g. two co-authors with swapped ORCIDs), so nodes are merged only when their *names are compatible*:
    within an ORCID group, nodes are linked when their name overlap >= `min_overlap`; a linked component is merged only
    if every pair in it is compatible. Incompatible nodes stay separate. The canonical node is the OpenAlex id with
    the most papers.
    """
    from itertools import combinations

    from .verify_metadata import author_overlap
    info: dict[str, dict] = {}
    for w in works:
        seen: set[str] = set()
        for a in w.get("authorships") or []:
            au = a.get("author") or {}
            aid, id_source = author_node_id(au)
            if aid is None or aid in seen:
                continue
            seen.add(aid)
            orc = (au.get("orcid") or "").rsplit("/", 1)[-1] or None
            d = info.setdefault(aid, {"orcid": orc, "name": clean_text(au.get("display_name") or a.get("raw_author_name")),
                                      "id_source": id_source, "n": 0})
            d["n"] += 1
            d["orcid"] = d["orcid"] or orc
    by_orcid: dict[str, list[str]] = defaultdict(list)
    for nid, d in info.items():
        if d["orcid"]:
            by_orcid[d["orcid"]].append(nid)

    def compatible(x: str, y: str) -> bool:
        return (author_overlap([info[x]["name"] or ""], [info[y]["name"] or ""]) or 0) >= min_overlap

    canon: dict[str, str] = {}
    for ids in by_orcid.values():
        if len(ids) < 2:
            continue
        parent = {i: i for i in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for x, y in combinations(ids, 2):
            if compatible(x, y):
                parent[find(x)] = find(y)
        comps: dict[str, list[str]] = defaultdict(list)
        for i in ids:
            comps[find(i)].append(i)
        for comp in comps.values():
            if len(comp) < 2 or not all(compatible(x, y) for x, y in combinations(comp, 2)):
                continue
            rep = max(comp, key=lambda i: (info[i]["id_source"] == "openalex", info[i]["n"], i))
            canon.update({i: rep for i in comp if i != rep})
    return canon


def _hierarchy_levels(t: dict) -> list[tuple[str, str, str]]:
    """topic dict -> [(node_id, name, level)] from topic up to domain."""
    out = [(schema.make_id("Topic", short_id(t["id"])), t["display_name"], "topic")]
    for level, prefix in (("subfield", "SF"), ("field", "F"), ("domain", "D")):
        if t.get(level) and t[level].get("id"):
            num = t[level]["id"].rstrip("/").rsplit("/", 1)[-1]
            out.append((schema.make_id("Topic", prefix + num), t[level]["display_name"], level))
    return out


class _EdgeBuffer:
    def __init__(self):
        self.rows: dict[tuple[str, str, str], dict] = {}

    def add(self, src: str, rel: str, dst: str, weight: float = 1.0, **props) -> None:
        key = (src, rel, dst)
        if key not in self.rows:  # first occurrence wins; duplicates are dropped
            self.rows[key] = {"src_id": src, "rel": rel, "dst_id": dst, "src_type": schema.type_of(src),
                              "dst_type": schema.type_of(dst), "weight": float(weight), "source": SOURCE,
                              "props": json.dumps(props, ensure_ascii=False) if props else "{}"}

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.rows.values()), columns=list(schema.EDGE_COLUMNS))


def _nodes_frame(node_type: str, rows: Iterable[dict]) -> pd.DataFrame:
    cols = list(schema.NODE_BASE_COLUMNS) + list(schema.NODE_TYPES[node_type].properties)
    return pd.DataFrame(list(rows), columns=cols)


def build_tables(works: list[dict], corpus_units: dict[str, list[str]] | None, gcfg: dict,
                 verification: dict[str, dict] | None = None,
                 alt_ids: dict[str, list[str]] | None = None,
                 abstract_fixes: dict[str, tuple[str, str | None]] | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """``verification``: paper id -> row of verification.parquet (status, new_title, new_abstract, new_year).
    ``alt_ids``: canonical paper id -> merged version ids; citations to those are redirected."""
    corpus_units = corpus_units or {}
    abstract_fixes = abstract_fixes or {}
    verification = verification or {}
    alt_ids = alt_ids or {}
    to_canonical = {a: c for c, alts in alt_ids.items() for a in alts}
    corpus_ids = {short_id(w["id"]) for w in works}
    max_topics = gcfg.get("max_topics_per_paper", 3)
    min_score = gcfg.get("min_topic_score", 0.0)
    hierarchy = gcfg.get("topic_hierarchy", True)
    internal_only = gcfg.get("citations", "internal") == "internal"

    author_canon = author_canonical_map(works, gcfg.get("author_merge_min_overlap", 0.5)) if gcfg.get("author_merge") else {}
    author_aliases: dict[str, list[str]] = defaultdict(list)
    for alias, rep in author_canon.items():
        author_aliases[rep].append(alias)

    edges = _EdgeBuffer()
    papers, authors, insts, venues, topics = [], {}, {}, {}, {}
    count = {k: defaultdict(int) for k in ("author", "inst", "venue", "topic")}
    affil: dict[tuple[str, str], list[int]] = defaultdict(list)

    for w in works:
        wid = short_id(w["id"])
        pid = schema.make_id("Paper", wid)
        v = verification.get(wid) or {}
        title = clean_text(w.get("title") or w.get("display_name"))
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        if v.get("status") == "repaired":  # OpenAlex text belonged to another record
            title, abstract = v["new_title"], v.get("new_abstract")
        if wid in abstract_fixes:  # `abstracts` step: the abstract belonged to another work
            abstract = abstract_fixes[wid][1]
        year = int(v["new_year"]) if pd.notna(v.get("new_year")) else w.get("publication_year")

        # venue
        venue_id = None
        src = ((w.get("primary_location") or {}).get("source") or {})
        if src.get("id"):
            venue_id = schema.make_id("Venue", short_id(src["id"]))
            venues.setdefault(venue_id, {"node_id": venue_id, "node_type": "Venue",
                                         "name": clean_text(src.get("display_name")), "source": SOURCE,
                                         "source_id": short_id(src["id"]), "venue_type": src.get("type"),
                                         "issn_l": src.get("issn_l")})
            count["venue"][venue_id] += 1
            edges.add(pid, "PUBLISHED_AT", venue_id)

        # authors & institutions
        paper_authors: set[str] = set()
        for a in w.get("authorships") or []:
            aid, id_source = author_node_id(a.get("author") or {})
            if aid is not None:
                aid = author_canon.get(aid, aid)      # persons merged by ORCID (+ compatible name)
            if aid is None or aid in paper_authors:
                continue
            paper_authors.add(aid)
            au = a.get("author") or {}
            authors.setdefault(aid, {"node_id": aid, "node_type": "Author",
                                     "name": clean_text(au.get("display_name") or a.get("raw_author_name")),
                                     "source": SOURCE, "source_id": short_id(au.get("id")),
                                     "orcid": (au.get("orcid") or "").rsplit("/", 1)[-1] or None,
                                     "id_source": id_source, "alt_ids": sorted(author_aliases.get(aid, []))})
            count["author"][aid] += 1
            edges.add(aid, "AUTHORED", pid, author_position=a.get("author_position"),
                      is_corresponding=bool(a.get("is_corresponding")))
            for inst in a.get("institutions") or []:
                if not inst.get("id"):
                    continue
                iid = schema.make_id("Institution", short_id(inst["id"]))
                insts.setdefault(iid, {"node_id": iid, "node_type": "Institution",
                                       "name": clean_text(inst.get("display_name")), "source": SOURCE,
                                       "source_id": short_id(inst["id"]),
                                       "ror": (inst.get("ror") or "").rsplit("/", 1)[-1] or None,
                                       "country_code": inst.get("country_code"), "inst_type": inst.get("type")})
                affil[(aid, iid)].append(year)
        for iid in {schema.make_id("Institution", short_id(i["id"]))
                    for a in w.get("authorships") or [] for i in a.get("institutions") or [] if i.get("id")}:
            count["inst"][iid] += 1

        # topics
        primary = (w.get("primary_topic") or {}).get("id")
        kept = [t for t in (w.get("topics") or [])[:max_topics]
                if t.get("id") and (t["id"] == primary or (t.get("score") or 0) >= min_score)]
        paper_topic_nodes: set[str] = set()  # every hierarchy level counts the paper once
        for t in kept:
            levels = _hierarchy_levels(t) if hierarchy else _hierarchy_levels(t)[:1]
            for nid, name, level in levels:
                topics.setdefault(nid, {"node_id": nid, "node_type": "Topic", "name": name, "source": SOURCE,
                                        "source_id": nid.split(":", 1)[1], "level": level})
                paper_topic_nodes.add(nid)
            for (child, *_), (parent, *_) in zip(levels, levels[1:]):
                edges.add(child, "SUBTOPIC_OF", parent)
            edges.add(pid, "HAS_TOPIC", levels[0][0], weight=t.get("score") or 1.0, is_primary=t["id"] == primary)
        for nid in paper_topic_nodes:
            count["topic"][nid] += 1

        # citations
        refs = [short_id(r) for r in w.get("referenced_works") or []]
        # a citation to the preprint of a corpus paper counts for the merged (canonical) paper
        internal = list(dict.fromkeys(c for r in refs if (c := to_canonical.get(r, r)) in corpus_ids and c != wid))
        for r in (internal if internal_only else refs):
            edges.add(pid, "CITES", schema.make_id("Paper", r))

        oa = w.get("open_access") or {}
        papers.append({
            "node_id": pid, "node_type": "Paper", "name": title,
            "source": SOURCE, "source_id": wid,
            "doi": None if v.get("status") == "doi_mismatch"
            else (w.get("doi") or "").replace("https://doi.org/", "") or None,
            "year": year, "publication_date": w.get("publication_date"), "work_type": w.get("type"),
            "language": w.get("language"), "abstract": abstract,
            "cited_by_count": w.get("cited_by_count"), "fwci": w.get("fwci"),
            "referenced_works_count": w.get("referenced_works_count"), "n_refs_internal": len(internal),
            "is_oa": oa.get("is_oa"), "oa_url": oa.get("oa_url"), "venue_id": venue_id,
            "primary_topic_id": schema.make_id("Topic", short_id(primary)) if primary else None,
            "referenced_works": refs, "corpus_units": list(corpus_units.get(wid, [])),
            "metadata_status": v.get("status", "unverified"),
            "alt_ids": list(alt_ids.get(wid, [])),
        })

    for (aid, iid), years in affil.items():
        ys = [y for y in years if y]
        edges.add(aid, "AFFILIATED_WITH", iid, weight=len(years),
                  first_year=min(ys) if ys else None, last_year=max(ys) if ys else None)

    for d, key in ((authors, "author"), (insts, "inst"), (venues, "venue"), (topics, "topic")):
        for nid, row in d.items():
            row["n_papers"] = count[key].get(nid, 0)

    nodes = {
        "Paper": _nodes_frame("Paper", papers),
        "Author": _nodes_frame("Author", authors.values()),
        "Institution": _nodes_frame("Institution", insts.values()),
        "Venue": _nodes_frame("Venue", venues.values()),
        "Topic": _nodes_frame("Topic", topics.values()),
    }
    edge_df = edges.frame()
    if not internal_only:
        # external citations point at papers outside the corpus; keep them as stubs
        known = set(nodes["Paper"]["node_id"])
        stubs = sorted(set(edge_df.loc[edge_df.rel == "CITES", "dst_id"]) - known)
        nodes["Paper"] = pd.concat([nodes["Paper"], _nodes_frame("Paper", (
            {"node_id": s, "node_type": "Paper", "name": None, "source": SOURCE,
             "source_id": s.split(":", 1)[1]} for s in stubs))], ignore_index=True)
    return nodes, edge_df


def load_corpus_works(cfg: Config, include_alt: bool = False) -> tuple[list[dict], dict[str, list[str]]]:
    """Raw OpenAlex records of the corpus papers (and, with include_alt, of their merged versions).
    A merged paper may live in another unit's file than the one that selected it, so all plan units
    are scanned."""
    corpus = read_parquet(cfg.paths.processed / "corpus.parquet")
    wanted = set(corpus["paper_id"])
    if include_alt and "alt_ids" in corpus:
        wanted |= {a for alts in corpus["alt_ids"] for a in alts}
    units = {pid: list(u) for pid, u in zip(corpus["paper_id"], corpus["corpus_units"])}
    works: dict[str, dict] = {}
    for unit in plan_units(cfg):
        for w in iter_jsonl(cfg.paths.openalex_raw / "works" / f"{unit.key}.jsonl"):
            wid = short_id(w["id"])
            if wid in wanted and wid not in works:
                works[wid] = w
    missing = set(corpus["paper_id"]) - set(works)
    if missing:
        raise RuntimeError(f"{len(missing)} corpus papers missing from raw store, e.g. {next(iter(missing))}")
    order = list(corpus["paper_id"]) + sorted(set(works) - set(corpus["paper_id"]))
    return [works[pid] for pid in order], units


def load_alt_ids(cfg: Config) -> dict[str, list[str]]:
    corpus = read_parquet(cfg.paths.processed / "corpus.parquet")
    if "alt_ids" not in corpus:
        return {}
    return {pid: list(a) for pid, a in zip(corpus["paper_id"], corpus["alt_ids"]) if len(a)}


def run(cfg: Config) -> dict:
    works, units = load_corpus_works(cfg)
    ver = load_verification(cfg)
    ver_map = {r["paper_id"]: r for r in ver.to_dict("records")} if ver is not None else {}
    nodes, edges = build_tables(works, units, cfg.get("graph", {}), ver_map, load_alt_ids(cfg), load_abstract_fixes(cfg))
    schema.validate(nodes, edges).raise_if_failed()
    out = cfg.paths.processed
    for tname, df in nodes.items():
        write_parquet(df, out / f"{schema.NODE_TYPES[tname].table}.parquet")
    write_parquet(edges, out / "edges.parquet")
    report = {"nodes": {k: len(v) for k, v in nodes.items()},
              "edges": edges["rel"].value_counts().to_dict()}
    write_json(out / "build_report.json", report)
    log.info("build: %s", report)
    return report
