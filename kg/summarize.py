"""Community summaries: the evidence given to the LLM, the prompt, and the output schema.

Evidence per community (everything is computed from the graph, nothing is looked up elsewhere):
  * size, median year, OpenAlex topics (as counted in communities.parquet)
  * characteristic entities per type from the LLM extraction, aggregated over SYNONYM_OF clusters (so "RAG" and
    "retrieval-augmented generation" count once); ranked by count x ln(lift), lift = share inside the community / share in the corpus,
    so corpus-wide staples ("accuracy") rank low
  * paper titles: the `n_central` closest to the community centroid (bge title+abstract embeddings) + `n_random` random members
  * for a level-0 community: the names and sizes of its level-1 children

The titles shown to the model are returned as `shown`; evaluation uses only members that were NOT shown.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROMPT_VERSION = "v1"
ENTITY_QUOTA = {"task": 10, "method": 12, "model": 8, "dataset": 8, "metric": 6}
NODE_TO_TYPE = {"Task": "task", "Method": "method", "Model": "model", "Dataset": "dataset", "Metric": "metric"}
TABLES = {"task": "tasks", "method": "methods", "model": "models", "dataset": "datasets", "metric": "metrics"}

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["name", "summary", "keywords", "subthemes", "is_mixed"],
          "properties": {"name": {"type": "string"}, "summary": {"type": "string"}, "keywords": {"type": "array", "items": {"type": "string"}},
                         "subthemes": {"type": "array", "items": {"type": "string"}}, "is_mixed": {"type": "boolean"}}}

_V1 = """You write the label and summary of one community of research papers (AI / computer science) found by graph clustering.
You get: its size, median year, OpenAlex topics, characteristic methods / tasks / models / datasets / metrics (with the number of member papers using each),
example paper titles (the closest to the community centre, then random members), and for a top-level community the names of its sub-communities.
Use ONLY this evidence. Do not add facts, names or numbers that are not in it.

Output
- name: 3-8 words, specific to this community (not "AI research", not a list of keywords). Prefer the research problem or technique that unites the papers.
- summary: 2-3 sentences, at most 70 words. What the papers study, the dominant techniques / tasks / datasets (only ones in the evidence), and what sets this community apart from neighbouring areas of AI.
- keywords: 6-8 short phrases taken from the evidence.
- subthemes: 2-4 short phrases for the main strands inside the community.
- is_mixed: true when the evidence shows several unrelated strands and no single theme explains most of the community (say so in the summary); false otherwise.
Notes: OpenAlex topic "Topic Modeling" is a catch-all label and says little. Entity counts are numbers of member papers."""

_V2 = _V1.replace("""- summary: 2-3 sentences, at most 70 words. What the papers study, the dominant techniques / tasks / datasets (only ones in the evidence), and what sets this community apart from neighbouring areas of AI.""",
"""- summary: 2-3 sentences, at most 70 words. What the papers study and the dominant techniques / tasks / datasets (only ones in the evidence). End with one sentence on how it differs from its closest communities (`closest_communities` in the evidence) when the evidence shows a real difference; otherwise leave the contrast out. Never write a generic contrast ("stands out for integrating ...", "unlike more traditional research").""").replace(
"""- is_mixed: true when the evidence shows several unrelated strands and no single theme explains most of the community (say so in the summary); false otherwise.""",
"""- is_mixed: true when the sub-communities, entities or titles split into two or more strands that researchers would not call one field (for example peer review and web-search behaviour, or LLM evaluation and code generation), so that no single theme explains most of the community; say so in the summary. false when one theme clearly dominates.""").replace(
"Notes: OpenAlex","Notes: `closest_communities` are the communities most linked to this one (shown with their main OpenAlex topics and TF-IDF terms), not part of this community. OpenAlex")
PROMPTS = {"v1": _V1, "v2": _V2}


def input_key(payload: dict, version: str) -> str:
    return hashlib.sha1((version + json.dumps(payload, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()[:24]


def load_graph_tables(processed: Path) -> dict:
    comm = pd.read_parquet(processed / "communities.parquet")
    edges = pd.read_parquet(processed / "edges_comm.parquet")
    papers = pd.read_parquet(processed / "papers.parquet")[["node_id", "name", "year"]]
    z = np.load(processed / "embeddings.npz", allow_pickle=False)
    vecs = dict(zip(z["ids"].tolist(), z["vecs"]))
    ext = pd.read_parquet(processed / "edges_extract.parquet")
    ext = ext[(ext.src_type == "Paper") & ext.dst_type.isin(NODE_TO_TYPE)][["src_id", "dst_id", "dst_type"]].drop_duplicates()
    links = pd.concat([pd.read_parquet(f, columns=["src_id", "dst_id", "rel", "weight"]) for f in sorted(processed.glob("edges*.parquet"))], ignore_index=True)
    links = links[links.rel.isin(["CITES", "SIMILAR_TO"]) & links.src_id.str.startswith("paper:") & links.dst_id.str.startswith("paper:")]
    names = {}
    for t in TABLES.values():
        f = processed / f"{t}_ext.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            names.update(dict(zip(d.node_id, d.name)))
    canon = {}
    cf = processed / "entity_clusters.parquet"
    if cf.exists():
        c = pd.read_parquet(cf)
        canon = dict(zip(c.node_id, c.canonical_id))
    return {"comm": comm, "edges": edges, "papers": papers, "vecs": vecs, "ext": ext, "names": names, "canon": canon, "links": links}


def membership(edges: pd.DataFrame) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for r in edges.itertuples():
        out[r.src_id][json.loads(r.props)["level"]] = r.dst_id.split(":", 1)[1]
    return out


def neighbours(t: dict, mem: dict, k: int = 3, min_share: float = 0.05) -> dict[str, list[tuple[str, float]]]:
    """community -> [(other community, share of its cross-paper link weight that goes there)], same level, by CITES + SIMILAR_TO weight."""
    w: dict[tuple[str, str], float] = defaultdict(float)
    for r in t["links"].itertuples():
        a, b = mem.get(r.src_id, {}), mem.get(r.dst_id, {})
        for level in (0, 1):
            if level in a and level in b and a[level] != b[level]:
                w[(a[level], b[level])] += float(r.weight)
                w[(b[level], a[level])] += float(r.weight)
    tot: dict[str, float] = defaultdict(float)
    for (a, _), x in w.items():
        tot[a] += x
    out: dict[str, list] = defaultdict(list)
    for (a, b), x in w.items():
        if x / tot[a] >= min_share:
            out[a].append((b, x / tot[a]))
    return {a: sorted(v, key=lambda y: -y[1])[:k] for a, v in out.items()}


def describe_old(row) -> str:
    """Neutral description of a community that does not depend on LLM output: main OpenAlex topics + TF-IDF terms."""
    kws = row["tfidf_keywords"] if "tfidf_keywords" in row.index and row["tfidf_keywords"] is not None else row["keywords"]
    return f"{'; '.join(t.rsplit(' (', 1)[0] for t in list(row['top_topics'])[:2])} | terms: {', '.join(list(kws)[:5])}"


def build_inputs(t: dict, n_central: int = 8, n_random: int = 4, seed: int = 7, use_entities: bool = True, use_neighbours: bool = True) -> dict[str, dict]:
    """community id (e.g. "L1-3") -> {"payload": <what the model sees>, "shown": [paper ids], "members": [paper ids], "level": int}"""
    comm, papers, vecs = t["comm"], t["papers"], t["vecs"]
    title = dict(zip(papers.node_id, papers.name))
    mem = membership(t["edges"])
    members: dict[str, list[str]] = defaultdict(list)
    for pid, lv in mem.items():
        for level, cid in lv.items():
            members[cid].append(pid)
    # entity usage per paper, aggregated over synonym clusters
    canon, names = t["canon"], t["names"]
    ent_of: dict[str, set] = defaultdict(set)
    for r in t["ext"].itertuples():
        ent_of[r.src_id].add((NODE_TO_TYPE[r.dst_type], canon.get(r.dst_id, r.dst_id)))
    corpus = Counter(e for s in ent_of.values() for e in s)
    n_corpus = max(1, len(ent_of))
    # children of level-0 communities
    parent: dict[str, Counter] = defaultdict(Counter)
    for lv in mem.values():
        if 0 in lv and 1 in lv:
            parent[lv[1]][lv[0]] += 1
    children: dict[str, list[str]] = defaultdict(list)
    for child, c in parent.items():
        children[c.most_common(1)[0][0]].append(child)
    rows = comm.set_index(comm.node_id.str.split(":", n=1).str[1])
    nb = neighbours(t, mem) if use_neighbours else {}
    out = {}
    for cid, row in rows.iterrows():
        ms = sorted(members[cid])
        mat = np.stack([vecs[p] for p in ms if p in vecs])
        have = [p for p in ms if p in vecs]
        cen = mat.mean(0)
        cen /= np.linalg.norm(cen) or 1.0
        order = np.argsort(-(mat @ cen))
        central = [have[i] for i in order[:n_central]]
        rest = [p for p in have if p not in set(central)]
        rng = np.random.default_rng(seed + sum(map(ord, cid)))      # deterministic per community (hash() is salted per process)
        rand = [rest[i] for i in rng.choice(len(rest), size=min(n_random, len(rest)), replace=False)] if rest else []
        payload = {"level": int(row["level"]), "papers": len(ms), "median_year": None if pd.isna(row["median_year"]) else int(row["median_year"]),
                   "openalex_topics": list(row["top_topics"])[:5]}
        if use_entities:
            cnt = Counter(e for p in ms for e in ent_of.get(p, ()))
            ents: dict[str, list[str]] = {}
            for typ, quota in ENTITY_QUOTA.items():
                scored = []
                for (ty, eid), c in cnt.items():
                    if ty != typ or c < 3:
                        continue
                    lift = (c / len(ms)) / (corpus[(ty, eid)] / n_corpus)
                    if lift > 1.2:
                        scored.append((c * math.log(lift), c, eid))
                scored.sort(reverse=True)
                ents[typ + "s"] = [f"{names.get(eid, eid)} ({c})" for _, c, eid in scored[:quota]]
            payload["characteristic_entities"] = {k: v for k, v in ents.items() if v}
        if use_neighbours and cid in nb:
            payload["closest_communities"] = [f"{describe_old(rows.loc[o])} ({share:.0%} of links)" for o, share in nb[cid]]
        payload["example_titles_central"] = [title[p] for p in central]
        payload["example_titles_random"] = [title[p] for p in rand]
        out[cid] = {"payload": payload, "shown": central + rand, "members": ms, "level": int(row["level"]), "children": sorted(children.get(cid, [])), "child_sizes": {c: int(rows.loc[c, "n_papers"]) for c in children.get(cid, [])}}
    return out
