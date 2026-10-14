"""Step `similar`: SIMILAR_TO edges from title+abstract embeddings.

Every paper gets its k nearest neighbours by cosine similarity (above `min_similarity`), so even
papers nobody cites are connected by content. A pair is stored once (src < dst) with
weight = similarity; treat the edge as undirected.

Embeddings are cached per (model, text) in data/raw/embeddings/<model>.npz, so scaling the corpus
only embeds the new papers.
Output: processed/<run>/edges_sim.parquet, embeddings.npz (this run's papers, for later steps)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re

import numpy as np
import pandas as pd

from .. import schema
from ..config import Config
from ..io import read_parquet, write_json, write_parquet

log = logging.getLogger(__name__)
SOURCE = "embedding"


def paper_text(title: str | None, abstract: str | None) -> str:
    return f"{title or ''}. {abstract or ''}".strip()


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:20]


def pin_gpu(gpu: int) -> None:
    """Make only GPU `gpu` visible to this process, before torch initialises CUDA.
    Other GPUs then do not exist for us, so they cannot be touched by accident."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly 1 visible GPU (physical {gpu}), found "
                           f"{torch.cuda.device_count()}: CUDA was initialised before pinning")


def embed_texts(texts: list[str], ecfg: dict) -> np.ndarray:
    pin_gpu(ecfg.get("gpu", 0))
    from sentence_transformers import SentenceTransformer  # heavy import: only when needed
    model = SentenceTransformer(ecfg["model"], device="cuda")  # = the single visible GPU
    model.max_seq_length = ecfg.get("max_seq_length", 512)
    return model.encode(texts, batch_size=ecfg.get("batch_size", 128), normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def cached_embeddings(cfg: Config, texts: list[str], ecfg: dict) -> np.ndarray:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ecfg["model"])
    path = cfg.paths.raw / "embeddings" / f"{slug}.npz"
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    keys = [_key(t) for t in texts]
    todo = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items())
    log.info("similar: %d texts, %d cached, %d to embed with %s", len(texts), len(texts) - len(todo),
             len(todo), ecfg["model"])
    if todo:
        vecs = embed_texts([t for _, t in todo], ecfg)
        cache.update({k: v for (k, _), v in zip(todo, vecs)})
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(list(cache)), vecs=np.stack(list(cache.values())))
        os.replace(tmp, path)
    return np.stack([cache[k] for k in keys])


def knn_pairs(emb: np.ndarray, k: int, min_sim: float, block: int = 2048) -> list[tuple[int, int, float, int]]:
    """(i, j, sim, rank) for the k nearest neighbours of each row (cosine; rows are unit vectors)."""
    n = len(emb)
    out = []
    for s in range(0, n, block):
        sims = emb[s:s + block] @ emb.T
        sims[np.arange(sims.shape[0]), np.arange(s, s + sims.shape[0])] = -1  # no self-neighbour
        top = np.argpartition(-sims, kth=min(k, n - 1) - 1, axis=1)[:, :k]
        for r in range(top.shape[0]):
            order = top[r][np.argsort(-sims[r, top[r]])]
            for rank, j in enumerate(order, 1):
                if sims[r, j] >= min_sim:
                    out.append((s + r, int(j), float(sims[r, j]), rank))
    return out


def run(cfg: Config) -> dict:
    ecfg = cfg.get("embedding", {})
    papers = read_parquet(cfg.paths.processed / "papers.parquet")
    papers = papers[papers["name"].notna()].reset_index(drop=True)
    texts = [paper_text(t, a) for t, a in zip(papers["name"], papers["abstract"])]
    emb = cached_embeddings(cfg, texts, ecfg)
    np.savez(cfg.paths.processed / "embeddings.npz", ids=papers["node_id"].to_numpy().astype(str), vecs=emb)

    k, min_sim = ecfg.get("k", 10), ecfg.get("min_similarity", 0.0)
    best: dict[tuple[str, str], tuple[float, int]] = {}
    ids = papers["node_id"].tolist()
    for i, j, sim, rank in knn_pairs(emb, k, min_sim):
        a, b = sorted((ids[i], ids[j]))
        if (a, b) not in best or rank < best[(a, b)][1]:
            best[(a, b)] = (sim, rank)
    rows = [{"src_id": a, "rel": "SIMILAR_TO", "dst_id": b, "src_type": "Paper", "dst_type": "Paper",
             "weight": round(sim, 5), "source": SOURCE,
             "props": json.dumps({"model": ecfg["model"], "rank": rank})}
            for (a, b), (sim, rank) in best.items()]
    edges = pd.DataFrame(rows, columns=list(schema.EDGE_COLUMNS))
    write_parquet(edges, cfg.paths.processed / "edges_sim.parquet")

    deg = pd.concat([edges.src_id, edges.dst_id]).value_counts()
    w = edges["weight"]
    report = {"papers": len(papers), "model": ecfg["model"], "k": k, "min_similarity": min_sim,
              "edges": len(edges), "papers_with_neighbours": int(deg.shape[0]),
              "mean_degree": round(float(deg.mean()), 2) if len(deg) else 0,
              "similarity_quantiles": {q: round(float(w.quantile(q)), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
              if len(w) else {}}
    write_json(cfg.paths.processed / "similar_report.json", report)
    log.info("similar: %s", report)
    return report
