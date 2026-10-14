"""Paper search over a built run: dense, lexical, graph (HippoRAG 2-style PPR) and community-routed (GraphRAG-style).

  python -m kg.retrieve "graph-based retrieval-augmented generation" [--system hybrid] [--k 10] [--latest] [--since 2025]

Systems (parameters are fixed here, before any evaluation query was run):
  lexical        TF-IDF cosine on title + abstract
  dense          bge-base cosine (query instruction) on title + abstract   = "naive RAG" retrieval
  graph          Personalized PageRank on the Paper-Entity(+CITES) graph, reset = entities matched to the query
  hybrid         PPR with reset = matched entities + top dense papers (HippoRAG 2: phrase and passage nodes in the reset)
  hybrid_nosyn   hybrid on the graph WITHOUT synonym linking (every extracted name is its own node)   - ablation
  hybrid_nocite  hybrid without CITES edges                                                             - ablation
  global         community-routed: the query picks the 2 closest level-1 community summaries, dense ranking inside them

Entity matching (the query side of HippoRAG): exact normalised n-gram match against every entity alias (a one-word n-gram
only when it looks like a name: BERT, LoRA, YOLOv8), plus the
10 nearest entity names by embedding (cosine >= ENTITY_MIN_COS). A seed is weighted by similarity / sqrt(#papers),
so specific entities ("GraphRAG", 31 papers) outweigh hubs ("RAG", 564). Metrics are not graph nodes (pure hubs).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import load_config
from .linking import link_key
from .steps.similar_papers import cached_embeddings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DAMPING = 0.5            # HippoRAG uses 0.5
PPR_ITERS = 40
ENTITY_TOPK = 10
ENTITY_MIN_COS = 0.80
DENSE_SEEDS = 30
PASSAGE_SHARE = 0.5      # share of the reset mass on dense paper seeds in `hybrid`
COMMUNITIES_ROUTED = 2
ENTITY_RELS = ("PROPOSES", "USES_METHOD", "USES_MODEL", "USES_DATASET", "ADDRESSES")
ENTITY_TYPES = ("Method", "Model", "Dataset", "Task")
TABLES = {"Method": "methods", "Model": "models", "Dataset": "datasets", "Task": "tasks"}
SYSTEMS = ("lexical", "dense", "graph", "hybrid", "hybrid_nosyn", "hybrid_nocite", "global")


def _name_like(tok: str) -> bool:
    """BERT, LoRA, YOLOv8, GPT-4, GraphRAG: upper case after the first letter, or a digit."""
    t = tok.strip(".,;:()[]\"'?")
    return len(t) >= 2 and (any(c.isdigit() for c in t) or any(c.isupper() for c in t[1:]))


@dataclass
class Graph:
    nodes: list[str]                 # papers first, then entities
    n_papers: int
    W: sp.csr_matrix                 # row-normalised transition matrix
    ent_ids: list[str]
    ent_vecs: np.ndarray
    ent_npapers: np.ndarray
    alias_key: dict[str, list[int]]  # normalised alias -> entity indices


class Searcher:
    def __init__(self, config: str = "configs/kg.yaml"):
        self.cfg = load_config(config)
        self.ecfg = self.cfg.get("embedding", {})
        d = self.cfg.paths.processed
        z = np.load(d / "embeddings.npz", allow_pickle=False)
        self.ids = z["ids"].tolist()
        self.V = z["vecs"]
        self.pidx = {p: i for i, p in enumerate(self.ids)}
        p = pd.read_parquet(d / "papers.parquet").set_index("node_id").loc[self.ids]
        self.titles, self.abstracts = p["name"].tolist(), p["abstract"].fillna("").tolist()
        self.dates = p["publication_date"].astype(str).tolist()
        self.years = p["year"].tolist()
        self.ext = pd.read_parquet(d / "edges_extract.parquet")
        self.ext = self.ext[self.ext.rel.isin(ENTITY_RELS) & self.ext.dst_type.isin(ENTITY_TYPES) & self.ext.src_id.isin(self.pidx)]
        names = {}
        for t in TABLES.values():
            f = d / f"{t}_ext.parquet"
            if f.exists():
                x = pd.read_parquet(f, columns=["node_id", "name"])
                names.update(dict(zip(x.node_id, x.name)))
        self.names = names
        cf = d / "entity_clusters.parquet"
        self.canon = dict(zip(*pd.read_parquet(cf)[["node_id", "canonical_id"]].T.values)) if cf.exists() else {}
        cites = pd.concat([pd.read_parquet(f, columns=["src_id", "rel", "dst_id"]) for f in sorted(d.glob("edges*.parquet"))])
        cites = cites[(cites.rel == "CITES") & cites.src_id.isin(self.pidx) & cites.dst_id.isin(self.pidx)]
        self.cites = list(zip(cites.src_id, cites.dst_id))
        self._graphs: dict[tuple[bool, bool], Graph] = {}
        self._lex = None
        self._comm = None

    # --- embeddings -----------------------------------------------------------------------------------------------------
    def embed(self, texts: list[str]) -> np.ndarray:
        """Texts seen before in this process come from memory: the on-disk cache is large, reading it per query is slow."""
        memo = self.__dict__.setdefault("_memo", {})
        todo = [t for t in dict.fromkeys(texts) if t not in memo]
        if todo:
            memo.update(zip(todo, cached_embeddings(self.cfg, todo, self.ecfg)))
        return np.stack([memo[t] for t in texts])

    def prepare(self, queries: list[str]) -> None:
        """Embed every query form (with and without the query instruction) in one call."""
        self.embed([QUERY_PREFIX + q for q in queries] + list(queries))

    def qvec(self, q: str) -> np.ndarray:
        return self.embed([QUERY_PREFIX + q])[0]

    # --- graph ------------------------------------------------------------------------------------------------------------
    def graph(self, synonyms: bool = True, citations: bool = True) -> Graph:
        key = (synonyms, citations)
        if key in self._graphs:
            return self._graphs[key]
        ent = (lambda x: self.canon.get(x, x)) if synonyms else (lambda x: x)
        pairs = {(r.src_id, ent(r.dst_id)) for r in self.ext.itertuples()}
        ent_ids = sorted({e for _, e in pairs})
        eidx = {e: self.n + i for i, e in enumerate(ent_ids)}
        rows, cols = [], []
        for pid, e in pairs:
            rows += [self.pidx[pid], eidx[e]]
            cols += [eidx[e], self.pidx[pid]]
        if citations:
            for a, b in self.cites:
                rows += [self.pidx[a], self.pidx[b]]
                cols += [self.pidx[b], self.pidx[a]]
        n = self.n + len(ent_ids)
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        A.data[:] = 1.0                                              # duplicate pairs count once
        deg = np.asarray(A.sum(1)).ravel()
        W = sp.diags(1.0 / np.maximum(deg, 1)) @ A
        npap = np.asarray((A[self.n:, :self.n] > 0).sum(1)).ravel()
        members = defaultdict(set)
        for node in self.names:
            members[ent(node)].add(node)
        alias_key: dict[str, list[int]] = defaultdict(list)
        for e in ent_ids:
            for m in members.get(e, {e}):
                k = link_key(self.names.get(m, m.split(":", 1)[-1]))
                if len(k) >= 3:
                    alias_key[k].append(eidx[e] - self.n)
        vecs = self.embed([self.names.get(e, e.split(":", 1)[-1]) for e in ent_ids])
        g = Graph([*self.ids, *ent_ids], self.n, W.tocsr(), ent_ids, vecs, npap, dict(alias_key))
        self._graphs[key] = g
        return g

    @property
    def n(self) -> int:
        return len(self.ids)

    def entity_seeds(self, g: Graph, q: str) -> dict[int, float]:
        seeds: dict[int, float] = {}
        toks = q.split()
        for i in range(len(toks)):
            for j in range(i + 1, min(len(toks), i + 6) + 1):
                if j == i + 1 and not _name_like(toks[i]):
                    continue                             # a single ordinary word ("learning") is not an entity mention
                for e in g.alias_key.get(link_key(" ".join(toks[i:j])), []):
                    seeds[e] = max(seeds.get(e, 0), 1.0)
        qe = self.embed([q])[0]
        sims = g.ent_vecs @ qe
        for e in np.argsort(-sims)[:ENTITY_TOPK]:
            if sims[e] >= ENTITY_MIN_COS:
                seeds[int(e)] = max(seeds.get(int(e), 0), float(sims[e]))
        return {e: w / np.sqrt(max(1, g.ent_npapers[e])) for e, w in seeds.items()}

    def ppr(self, g: Graph, reset: np.ndarray) -> np.ndarray:
        p = reset.copy()
        WT = g.W.T.tocsr()
        for _ in range(PPR_ITERS):
            p = (1 - DAMPING) * reset + DAMPING * (WT @ p)
        return p

    def graph_scores(self, q: str, passage_share: float, synonyms=True, citations=True) -> tuple[np.ndarray, dict]:
        g = self.graph(synonyms, citations)
        reset = np.zeros(len(g.nodes))
        seeds = self.entity_seeds(g, q)
        if seeds:
            tot = sum(seeds.values())
            for e, w in seeds.items():
                reset[g.n_papers + e] = (1 - passage_share) * w / tot
        if passage_share > 0 or not seeds:
            share = passage_share if seeds else 1.0
            ds = self.V @ self.qvec(q)
            top = np.argsort(-ds)[:DENSE_SEEDS]
            w = ds[top] - ds[top].min() + 1e-6
            reset[top] += share * w / w.sum()
        p = self.ppr(g, reset)
        return p[:g.n_papers], {"seeds": [g.ent_ids[e] for e in sorted(seeds, key=lambda e: -seeds[e])][:8]}

    # --- other systems ----------------------------------------------------------------------------------------------------
    def lexical_scores(self, q: str) -> np.ndarray:
        if self._lex is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, ngram_range=(1, 2))
            self._lex = (vec, vec.fit_transform([f"{t}. {a}" for t, a in zip(self.titles, self.abstracts)]))
        vec, M = self._lex
        return (M @ vec.transform([q]).T).toarray().ravel()

    def communities(self):
        if self._comm is None:
            d = self.cfg.paths.processed
            c = pd.read_parquet(d / "communities.parquet")
            c = c[(c.level == 1) & c.summary.notna()]
            text = [f"{r.name}. {r.summary} Keywords: {', '.join(list(r.keywords))}." for r in c.itertuples()]
            e = pd.read_parquet(d / "edges_comm.parquet")
            e = e[e.props.map(lambda x: json.loads(x)["level"]) == 1]
            members = defaultdict(list)
            for r in e.itertuples():
                if r.src_id in self.pidx:
                    members[r.dst_id].append(self.pidx[r.src_id])
            self._comm = (c.node_id.tolist(), self.embed(text), members, c.name.tolist())
        return self._comm

    def global_scores(self, q: str) -> tuple[np.ndarray, dict]:
        cids, S, members, cnames = self.communities()
        qv = self.qvec(q)
        top = np.argsort(-(S @ qv))[:COMMUNITIES_ROUTED]
        ds = self.V @ qv
        scores = ds - 10.0                                           # outside the routed communities: after every member
        for t in top:
            idx = members[cids[t]]
            scores[idx] = ds[idx]
        return scores, {"communities": [cnames[t] for t in top]}

    def search(self, q: str, system: str = "hybrid", k: int = 10) -> tuple[list[str], dict]:
        info: dict = {}
        if system == "lexical":
            s = self.lexical_scores(q)
        elif system == "dense":
            s = self.V @ self.qvec(q)
        elif system == "graph":
            s, info = self.graph_scores(q, 0.0)
        elif system == "hybrid":
            s, info = self.graph_scores(q, PASSAGE_SHARE)
        elif system == "hybrid_nosyn":
            s, info = self.graph_scores(q, PASSAGE_SHARE, synonyms=False)
        elif system == "hybrid_nocite":
            s, info = self.graph_scores(q, PASSAGE_SHARE, citations=False)
        elif system == "global":
            s, info = self.global_scores(q)
        else:
            raise ValueError(system)
        order = np.argsort(-s)[:k]
        return [self.ids[i] for i in order], info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--config", default="configs/kg.yaml")
    ap.add_argument("--system", default="hybrid", choices=SYSTEMS)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--latest", action="store_true", help="take the 50 most relevant, show the newest k")
    ap.add_argument("--since", type=int)
    a = ap.parse_args(argv)
    s = Searcher(a.config)
    ids, info = s.search(a.query, a.system, 200 if (a.latest or a.since) else a.k)
    rows = [(p, s.dates[s.pidx[p]], s.titles[s.pidx[p]]) for p in ids]
    if a.since:
        rows = [r for r in rows if int(r[1][:4]) >= a.since]
    if a.latest:
        rows = sorted(rows[:50], key=lambda r: r[1], reverse=True)
    for p, dte, t in rows[:a.k]:
        print(f"{dte[:10]}  {p.split(':')[1]}  {t}")
    if info:
        print("\n" + json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
