"""Entity linking: which extracted Method / Model / Dataset / Task / Metric nodes are the same entity?

Evidence tiers, strongest first. Nothing is merged destructively: the step writes SYNONYM_OF edges (alias -> canonical) and a
cluster table; nodes and their provenance stay as extracted.

  t1  same name after light normalisation: plural -> singular on the last word, hyphen / space / case, and one generic tail word
      per type ("F1 score" = "F1", "ImageNet dataset" = "ImageNet", "GPT-4 model" = "GPT-4")
  t2  corpus-wide acronym dictionary: "retrieval-augmented generation (RAG)" defined in many abstracts -> RAG = retrieval-augmented
      generation. Only acronyms with enough papers and one dominant expansion ("RS" = remote sensing / recommender systems is skipped)
  t3  name-embedding neighbours (bge, GPU 0), guarded: same numbers (GPT-3 != GPT-3.5), no hypernym/hyponym pairs
      ("object detection" vs "few-shot object detection"), no very short names

Same type only. Clusters are built by union-find in tier order with a size cap; the canonical node is the one used by most papers.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import extraction as x

log = logging.getLogger(__name__)

TYPES = ("method", "model", "dataset", "task", "metric")
TABLES = {"method": "methods", "model": "models", "dataset": "datasets", "task": "tasks", "metric": "metrics"}
TAILS = {"metric": {"score", "metric", "measure", "value"}, "method": {"algorithm", "approach", "method", "technique"},
         "model": {"model"}, "dataset": {"dataset", "data", "benchmark", "corpus", "set"}, "task": {"task", "problem"}}
_KEEP_S = {"bias", "analysis", "basis", "this", "corpus", "status", "focus", "bonus", "atlas", "canvas", "lens", "gas", "bus", "consensus", "thesis",
           "synthesis", "hypothesis", "diagnosis", "physics", "statistics", "mathematics", "dynamics", "robotics", "analytics", "graphics",
           "kinetics", "ethics", "news", "series", "species", "access", "process", "class", "loss", "pass", "success", "progress", "address"}
DEFAULTS = {"bag": True, "tiers": ["t1", "t2", "t3"], "acronym_min_papers": 2, "acronym_dominance": 0.7, "tail_min_rest": 4,
            "knn_k": 10, "embed_min_cos": 0.85, "embed_min_cos_nojudge": 0.97, "embed_min_len": 6, "max_cluster": 25, "profile_min": 0.0}


_SUFFIXES = ("ization", "isation", "izing", "ising", "ations", "ation", "ized", "ised", "ings", "ing", "ions", "ion", "ed")
_FILLER = {"level", "based"}                                    # "token-level F1" = "token F1", "LLM-based agents" = "LLM agents"
_STOP = {"of", "the", "a", "an", "and"}
_ORDERED = {"to", "from", "for", "vs", "versus", "by", "over", "under", "against", "with", "on", "in", "via"}   # direction matters: "image-to-text" != "text-to-image"


def singular(w: str) -> str:
    if len(w) <= 3 or w in _KEEP_S:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("sses", "shes", "ches", "xes", "zes")):
        return w[:-2]
    if w.endswith(("ss", "us", "is")):
        return w
    return w[:-1] if w.endswith("s") else w


def stem(w: str) -> str:
    """Deliberately light: nominalisation / gerund / plural / British spelling only (summarization = summarizing = summarisation)."""
    if any(c.isdigit() for c in w) or len(w) <= 4:
        return singular(w)
    w = re.sub(r"(?<=[a-z])our$", "or", w)
    w = re.sub(r"(?<=[a-z])is(?=(ation|ing|ed|e)\b)", "iz", w)
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return singular(w)


def _raw_tokens(name: str) -> list[str]:
    s = re.sub(r"[\u2122\u00ae\u00a9]", "", name)                   # trademark signs
    s = re.sub(r"\([^)]*\)", " ", unicodedata.normalize("NFKC", s))
    s = re.sub(r"(?<=\d)\.0\b", "", s)                               # GPT-4.0 = GPT-4
    return re.findall(r"[\w+#@]+", s)


def words(name: str) -> list[str]:
    return [t.lower() for t in _raw_tokens(name)]


def link_key(name: str, typ: str | None = None, tail_min_rest: int = 4, bag: bool = True) -> str:
    raw = _raw_tokens(name)
    if not raw:
        return ""
    toks = []
    for t in raw:
        acronym_plural = t.isupper() and len(t) <= 6 and t.endswith("S")          # "ANNS" is not the plural of "ANN"
        low = t.lower()
        toks.append(low if acronym_plural else stem(low) if bag else low)
    tails = TAILS.get(typ or "", ())
    while len(toks) > 1 and toks[-1] in tails and len("".join(toks[:-1])) >= tail_min_rest:
        toks.pop()
    if not bag:
        return "".join(toks)
    # "prediction of X" / "predicting X" -> "X prediction"; any other word order is kept ("generation-augmented retrieval" != "retrieval-augmented generation")
    if "of" in toks[1:] and toks[0] not in _STOP and not any(t in _ORDERED for t in toks):
        i = toks.index("of", 1)
        toks = toks[i + 1:] + toks[:i]
    elif len(raw[0]) > 5 and raw[0].lower().endswith("ing") and len(toks) > 1 and not any(t in _ORDERED for t in toks):
        toks = toks[1:] + toks[:1]
    keep = [t for t in toks if t not in _STOP]
    rest = [t for t in keep if t not in _FILLER]
    if len(rest) >= 2:                      # "policy-based" / "policy-level" stay apart; "LLM-based agents" = "LLM agents"
        keep = rest
    return "".join(keep)


# --- context ------------------------------------------------------------------------------------------------------------
@dataclass
class Ctx:
    nodes: pd.DataFrame                      # node_id, typ, name, n_papers
    papers_of: dict[str, set]                # node_id -> set of paper node ids
    paper_vecs: dict[str, np.ndarray] | None = None
    acronyms: dict[str, Counter] = field(default_factory=dict)   # short key -> Counter(long key -> n papers)
    titles: dict[str, str] = field(default_factory=dict)         # paper node id -> title (context for the judge)

    def titles_of(self, paper_ids, n: int = 1) -> list[str]:
        return [self.titles[p][:90] for p in sorted(paper_ids)[:n] if p in self.titles]


def load_ctx(processed: Path, with_vecs: bool = True) -> Ctx:
    frames = []
    for typ, table in TABLES.items():
        f = processed / f"{table}_ext.parquet"
        if f.exists():
            d = pd.read_parquet(f)[["node_id", "name", "n_papers"]]
            frames.append(d.assign(typ=typ))
    nodes = pd.concat(frames, ignore_index=True)
    edges = pd.read_parquet(processed / "edges_extract.parquet")
    edges = edges[edges.src_type == "Paper"][["src_id", "dst_id"]].drop_duplicates()
    papers_of = {k: set(v) for k, v in edges.groupby("dst_id").src_id}
    vecs = None
    if with_vecs and (processed / "embeddings.npz").exists():
        z = np.load(processed / "embeddings.npz", allow_pickle=False)
        vecs = dict(zip(z["ids"].tolist(), z["vecs"]))
    return Ctx(nodes, papers_of, vecs)


def build_acronyms(papers: pd.DataFrame) -> dict[str, Counter]:
    """Corpus-wide {short key: Counter(long key -> number of papers defining it)} from "Long Form (SF)" patterns."""
    table: dict[str, Counter] = defaultdict(Counter)
    for title, abstract in zip(papers["name"], papers["abstract"]):
        seen = set()
        for lf, sf in x.abbreviation_pairs(f"{title}. {abstract}"):
            pair = (link_key(sf), link_key(lf))
            if pair[0] and pair[1] and pair not in seen:
                seen.add(pair)
                table[pair[0]][pair[1]] += 1
    return table


# --- clustering ---------------------------------------------------------------------------------------------------------
class UnionFind:
    def __init__(self, sizes: dict):
        self.p = {k: k for k in sizes}
        self.size = dict(sizes)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b, cap: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb or self.size[ra] + self.size[rb] > cap:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _digits(name: str) -> list[str]:
    return sorted(re.findall(r"\d+(?:\.\d+)?", name))


def _hypernym_pair(a: str, b: str) -> bool:
    """One name's words are a strict subset of the other's ("object detection" / "few-shot object detection")."""
    sa, sb = {singular(w) for w in words(a)}, {singular(w) for w in words(b)}
    return sa != sb and (sa < sb or sb < sa)


def compute_links(ctx: Ctx, params: dict | None = None, embed=None, judge=None) -> dict:
    """`embed(list[str]) -> unit vectors` and `judge(list[pair dict]) -> list[bool]` are only called for tier t3."""
    P = {**DEFAULTS, **(params or {})}
    nodes = ctx.nodes
    # t1 groups: (type, link key)
    nodes = nodes.assign(lk=[link_key(n, t, P["tail_min_rest"], P["bag"]) for n, t in zip(nodes["name"], nodes["typ"])])
    nodes = nodes[nodes.lk != ""]
    group_of = {nid: (t, k) for nid, t, k in zip(nodes.node_id, nodes.typ, nodes.lk)}
    members: dict[tuple, list[str]] = defaultdict(list)
    for nid, g in group_of.items():
        members[g].append(nid)
    npap = dict(zip(nodes.node_id, nodes.n_papers))
    name_of = dict(zip(nodes.node_id, nodes.name))
    gpapers = {g: set().union(*(ctx.papers_of.get(n, set()) for n in m)) for g, m in members.items()}
    gname = {g: name_of[max(m, key=lambda n: (npap[n], -len(name_of[n])))] for g, m in members.items()}
    uf = UnionFind({g: 1 for g in members})
    unions: list[dict] = []
    stats = {"groups": len(members), "t1_merged_nodes": len(group_of) - len(members), "t2_pairs": 0, "t3_pairs": 0, "t2_unions": 0, "t3_unions": 0}

    def try_union(ga, gb, tier, score, why=""):
        if uf.union(ga, gb, P["max_cluster"]):
            unions.append({"a": ga, "b": gb, "tier": tier, "score": float(score), "why": why})
            return True
        return False

    # t2: acronym dictionary
    if "t2" in P["tiers"]:
        for sk, c in ctx.acronyms.items():
            total = sum(c.values())
            lk, n = c.most_common(1)[0]
            if total < P["acronym_min_papers"] or n / total < P["acronym_dominance"]:
                continue
            for typ in TYPES:
                ga, gb = (typ, sk), (typ, lk)
                if ga in members and gb in members:
                    stats["t2_pairs"] += 1
                    stats["t2_unions"] += try_union(ga, gb, "t2", n / total, f"{sk}={lk} ({n}/{total})")

    # t3: name-embedding candidates, verified by `judge` (an LLM) when given; without a judge only very close names pass the thresholds
    if "t3" in P["tiers"] and embed is not None:
        cands, seen_pairs = [], set()
        for typ in TYPES:
            seen_pairs.clear()
            gs = [g for g in members if g[0] == typ and len(g[1]) >= P["embed_min_len"]]
            if len(gs) < 2:
                continue
            vecs = embed([gname[g] for g in gs])
            for s in range(0, len(gs), 2048):
                sims = vecs[s:s + 2048] @ vecs.T
                sims[np.arange(sims.shape[0]), np.arange(s, s + sims.shape[0])] = -1
                top = np.argsort(-sims, axis=1)[:, :P["knn_k"]]
                for r in range(top.shape[0]):
                    i = s + r
                    for j in top[r]:            # sorted by similarity, so `break` at the first one below the threshold
                        sim = float(sims[r, j])
                        if sim < P["embed_min_cos"]:
                            break
                        key = (min(i, j), max(i, j))
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        a, b = gname[gs[i]], gname[gs[j]]
                        if _digits(a) == _digits(b) and (judge is not None or not _hypernym_pair(a, b)):
                            cands.append((sim, gs[i], gs[j]))
        cands.sort(key=lambda c: -c[0])
        stats["t3_pairs"] = len(cands)
        if judge is not None and cands:
            verdicts = judge([{"type": g1[0], "a": gname[g1], "b": gname[g2], "ctx_a": ctx.titles_of(gpapers[g1]), "ctx_b": ctx.titles_of(gpapers[g2])}
                              for _, g1, g2 in cands])
            cands = [c for c, v in zip(cands, verdicts) if v]
            stats["t3_judged_same"] = len(cands)
        elif judge is None:
            cands = [c for c in cands if c[0] >= P["embed_min_cos_nojudge"]]
        for sim, g1, g2 in cands:
            stats["t3_unions"] += try_union(g1, g2, "t3", sim, f"{gname[g1]} ~ {gname[g2]}")

    # canonical per cluster, and a spanning forest (the accepted unions) oriented from the canonical group
    comp: dict = defaultdict(list)
    for g in members:
        comp[uf.find(g)].append(g)
    adj: dict = defaultdict(list)
    for u in unions:
        adj[u["a"]].append((u["b"], u))
        adj[u["b"]].append((u["a"], u))
    clusters, attach = {}, {}
    for root, gs in comp.items():
        canon_g = max(gs, key=lambda g: (len(gpapers[g]), -len(gname[g]), g[1]))
        canon_node = max(members[canon_g], key=lambda n: (npap[n], -len(name_of[n]), n))
        seen, queue = {canon_g}, [canon_g]
        attach[canon_g] = None
        while queue:
            g = queue.pop(0)
            for h, u in adj[g]:
                if h not in seen:
                    seen.add(h)
                    attach[h] = u
                    queue.append(h)
        for g in gs:
            for n in members[g]:
                clusters[n] = {"canonical": canon_node, "group": g, "canonical_group": canon_g,
                               "tier": "t1" if g == canon_g else attach[g]["tier"], "score": 1.0 if g == canon_g else attach[g]["score"],
                               "why": "" if g == canon_g else attach[g]["why"]}
    return {"groups": members, "gpapers": gpapers, "gname": gname, "clusters": clusters, "unions": unions, "stats": stats, "npap": npap,
            "name_of": name_of, "typ_of": dict(zip(nodes.node_id, nodes.typ))}
