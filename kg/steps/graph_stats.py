"""Step `stats`: validation + descriptive statistics -> exports/<run>/stats.{json,md}."""
from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

from .. import schema
from ..config import Config
from ..io import write_json
from .export_graph import load_processed

log = logging.getLogger(__name__)


def _top(df: pd.DataFrame, col: str = "n_papers", n: int = 10) -> list[dict]:
    if df is None or df.empty or col not in df:
        return []
    return [{"name": r["name"], col: int(r[col])} for _, r in df.nlargest(n, col).iterrows()]


def compute_stats(nodes: dict[str, pd.DataFrame], edges: pd.DataFrame) -> dict:
    rep = schema.validate(nodes, edges)
    papers = nodes["Paper"]
    real = papers[papers["name"].notna()]  # excludes external citation stubs, if any
    n = len(real)

    cites = edges[edges.rel == "CITES"]
    cg = nx.DiGraph()
    cg.add_nodes_from(real["node_id"])
    cg.add_edges_from(zip(cites.src_id, cites.dst_id))
    wcc = sorted((len(c) for c in nx.weakly_connected_components(cg)), reverse=True)
    in_deg = cites.dst_id.value_counts()

    g = nx.Graph()
    for df in nodes.values():
        g.add_nodes_from(df["node_id"])
    g.add_edges_from(zip(edges.src_id, edges.dst_id))
    all_cc = sorted((len(c) for c in nx.connected_components(g)), reverse=True)

    degree = pd.Series(dict(g.degree()))
    per_type_deg = {t: round(float(degree.reindex(df["node_id"]).mean()), 2) for t, df in nodes.items() if len(df)}

    topics = nodes.get("Topic")
    authors = nodes.get("Author")
    most_cited = real.assign(internal_citations=real["node_id"].map(in_deg).fillna(0).astype(int)) \
                     .nlargest(10, "internal_citations")

    return {
        "validation": {"ok": rep.ok, "errors": rep.errors},
        "nodes": {t: len(df) for t, df in nodes.items()},
        "edges": edges["rel"].value_counts().to_dict(),
        "totals": {"nodes": int(sum(len(df) for df in nodes.values())), "edges": len(edges)},
        "mean_degree_by_type": per_type_deg,
        "graph_connected_components": {"count": len(all_cc), "largest": all_cc[0] if all_cc else 0,
                                       "largest_frac": round(all_cc[0] / max(1, g.number_of_nodes()), 4) if all_cc else 0},
        "papers": {
            "count": n,
            "abstract_coverage": round(real["abstract"].notna().mean(), 4) if n else 0,
            "open_access_frac": round(real["is_oa"].fillna(False).astype(bool).mean(), 4) if n else 0,
            "by_year": {int(k): int(v) for k, v in real["year"].value_counts().sort_index().items()},
            "by_type": real["work_type"].value_counts().to_dict(),
            # OpenAlex lacks reference lists for most arXiv preprints -> they have no outgoing CITES
            "with_reference_list_by_type": real.assign(has=real["referenced_works_count"].fillna(0) > 0)
                                               .groupby("work_type")["has"].mean().round(4).to_dict(),
            "median_cited_by_count": float(real["cited_by_count"].median()) if n else 0,
            "mean_authors_per_paper": round(len(edges[edges.rel == "AUTHORED"]) / max(1, n), 2),
        },
        "citations_internal": {
            "edges": len(cites),
            "papers_citing_any": int(cites.src_id.nunique()),
            "papers_cited_by_any": int(cites.dst_id.nunique()),
            "papers_isolated": int(sum(1 for c in nx.weakly_connected_components(cg) if len(c) == 1)),
            "largest_wcc": wcc[0] if wcc else 0,
            "largest_wcc_frac": round(wcc[0] / max(1, n), 4) if wcc else 0,
            "mean_internal_refs": round(len(cites) / max(1, n), 2),
            "by_source": cites["source"].value_counts().to_dict(),
        },
        "authors": {
            "id_source": authors["id_source"].value_counts().to_dict() if authors is not None else {},
        },
        "top": {
            "internally_cited_papers": [{"name": r["name"], "year": int(r["year"]),
                                         "internal_citations": int(r["internal_citations"])}
                                        for _, r in most_cited.iterrows()],
            "authors": _top(authors),
            "institutions": _top(nodes.get("Institution")),
            "venues": _top(nodes.get("Venue")),
            "topics": _top(topics[topics.level == "topic"]) if topics is not None else [],
            "subfields": _top(topics[topics.level == "subfield"]) if topics is not None else [],
            "countries": nodes["Institution"]["country_code"].value_counts().head(10).to_dict()
            if "Institution" in nodes else {},
            "communities": _top(nodes.get("Community")) if "Community" in nodes else [],
            "hf_models": _top(nodes.get("Model", pd.DataFrame()).dropna(subset=["hf_downloads"])
                              if "Model" in nodes else None, "hf_downloads"),
            "hf_datasets": _top(nodes.get("Dataset", pd.DataFrame()).dropna(subset=["hf_downloads"])
                                if "Dataset" in nodes else None, "hf_downloads"),
        },
    }


def to_markdown(s: dict) -> str:
    lines = [f"# KG statistics\n", f"Validation: **{'OK' if s['validation']['ok'] else 'FAILED'}**"]
    lines += [f"- {e}" for e in s["validation"]["errors"]]
    lines += ["", "| Node type | Count | Mean degree |", "|---|---:|---:|"]
    lines += [f"| {t} | {c:,} | {s['mean_degree_by_type'].get(t, 0)} |" for t, c in s["nodes"].items()]
    lines += ["", "| Edge type | Count |", "|---|---:|"]
    lines += [f"| {t} | {c:,} |" for t, c in s["edges"].items()]
    p, c = s["papers"], s["citations_internal"]
    lines += ["", "## Papers", f"- papers: {p['count']:,}, abstract coverage {p['abstract_coverage']:.1%}, "
              f"OA {p['open_access_frac']:.1%}, mean authors {p['mean_authors_per_paper']}",
              f"- by year: {p['by_year']}",
              f"- by type: {p['by_type']}; share with a reference list: {p['with_reference_list_by_type']}",
              "", "## Internal citation graph",
              f"- CITES edges {c['edges']:,} (by source: {c['by_source']}); mean internal refs/paper {c['mean_internal_refs']}",
              f"- papers citing >=1 corpus paper: {c['papers_citing_any']:,}; cited by >=1: {c['papers_cited_by_any']:,}",
              f"- isolated papers: {c['papers_isolated']:,}; largest WCC {c['largest_wcc']:,} ({c['largest_wcc_frac']:.1%})",
              "", f"Whole graph: {s['graph_connected_components']['count']:,} components, largest "
              f"{s['graph_connected_components']['largest_frac']:.1%} of nodes",
              f"Author id source: {s['authors']['id_source']}"]
    for k, rows in s["top"].items():
        if isinstance(rows, dict):
            lines += ["", f"## Top {k}", ", ".join(f"{a} ({b})" for a, b in rows.items())]
        elif rows:
            lines += ["", f"## Top {k}"] + [f"1. {r['name']} — " + ", ".join(f"{kk}={vv}" for kk, vv in r.items()
                                                                          if kk != "name") for r in rows]
    return "\n".join(lines) + "\n"


def run(cfg: Config) -> dict:
    nodes, edges = load_processed(cfg)
    s = compute_stats(nodes, edges)
    out = cfg.paths.exports
    write_json(out / "stats.json", s)
    (out / "stats.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "stats.md").write_text(to_markdown(s))
    log.info("stats: nodes=%s edges=%s valid=%s", s["totals"]["nodes"], s["totals"]["edges"], s["validation"]["ok"])
    if not s["validation"]["ok"]:
        raise ValueError("validation failed: " + "; ".join(s["validation"]["errors"]))
    return s
