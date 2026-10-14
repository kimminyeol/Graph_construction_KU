"""Step `export`: processed tables -> exports/<run>/.

  nodes.parquet / edges.parquet   unified tables (type-specific props as JSON)
  graph.graphml                   for Gephi / networkx / igraph
  neo4j/                          CSVs + import.sh for `neo4j-admin database import`
  schema.json, manifest.json      schema snapshot, config snapshot + counts
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd

from .. import schema
from ..config import Config
from ..io import read_json, read_parquet, write_json, write_parquet

log = logging.getLogger(__name__)


def load_processed(cfg: Config) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Every node table present in processed/<run>/ plus edges.parquet (base graph) and every
    edges_<step>.parquet written by later steps - new steps just add files."""
    nodes = {}
    for nt in schema.NODE_TYPES.values():
        # `<table>.parquet` plus `<table>_<tag>.parquet` (e.g. models_ext.parquet from the LLM extraction)
        parts = [pd.read_parquet(p) for p in [cfg.paths.processed / f"{nt.table}.parquet", *sorted(cfg.paths.processed.glob(f"{nt.table}_*.parquet"))] if p.exists()]
        parts = [d for d in parts if len(d)]
        if parts:
            df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
            nodes[nt.name] = df.drop_duplicates("node_id", keep="first")
    base = read_parquet(cfg.paths.processed / "edges.parquet")
    extra = [pd.read_parquet(p) for p in sorted(cfg.paths.processed.glob("edges_*.parquet"))]
    return nodes, pd.concat([base, *extra], ignore_index=True)


def _scalar(v):
    """Python scalar suitable for GraphML, or None to drop the attribute."""
    if v is None or isinstance(v, (list, tuple, dict, np.ndarray)):
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _props_json(row: pd.Series, cols: list[str]) -> str:
    out = {}
    for c in cols:
        v = row[c]
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out[c] = v
    return json.dumps(out, ensure_ascii=False)


def unified_nodes(nodes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for tname, df in nodes.items():
        extra = [c for c in df.columns if c not in schema.NODE_BASE_COLUMNS]
        f = df[list(schema.NODE_BASE_COLUMNS)].copy()
        f["props"] = [_props_json(r, extra) for _, r in df[extra].iterrows()] if extra else "{}"
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def to_networkx(nodes: dict[str, pd.DataFrame], edges: pd.DataFrame,
                include_abstract: bool = False) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for tname, df in nodes.items():
        drop = set() if include_abstract else {"abstract"}
        for rec in df.to_dict("records"):
            attrs = {k: s for k, v in rec.items() if k not in drop and k != "node_id"
                     and (s := _scalar(v)) is not None}
            g.add_node(rec["node_id"], **attrs)
    for rec in edges.to_dict("records"):
        attrs = {"rel": rec["rel"], "weight": float(rec["weight"])}
        for k, v in json.loads(rec["props"]).items():
            if (s := _scalar(v)) is not None:
                attrs[k] = s
        g.add_edge(rec["src_id"], rec["dst_id"], key=rec["rel"], **attrs)
    return g


def _neo4j_header(df: pd.DataFrame, id_col: str) -> tuple[pd.DataFrame, list[str]]:
    out, header = pd.DataFrame(index=df.index), []
    for c in df.columns:
        s = df[c]
        if c == id_col:
            out[c], h = s, f"{c}:ID"
        elif s.map(lambda v: isinstance(v, (list, np.ndarray))).any():
            out[c] = s.map(lambda v: ";".join(map(str, v)) if isinstance(v, (list, np.ndarray)) else "")
            h = f"{c}:string[]"
        elif pd.api.types.is_bool_dtype(s):
            out[c], h = s, f"{c}:boolean"
        elif pd.api.types.is_integer_dtype(s):
            out[c], h = s, f"{c}:long"
        elif pd.api.types.is_float_dtype(s):
            # nullable ints (e.g. year) arrive as float; keep them integral when they are
            whole = s.dropna().map(float.is_integer).all()
            out[c], h = (s.astype("Int64"), f"{c}:long") if whole else (s, f"{c}:double")
        else:
            out[c], h = s, c
        header.append(h)
    return out, header


def export_neo4j(nodes: dict[str, pd.DataFrame], edges: pd.DataFrame, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    args = []
    for tname, df in nodes.items():
        f, header = _neo4j_header(df, "node_id")
        f.to_csv(out_dir / f"nodes_{tname}.csv", index=False, header=header)
        args.append(f"--nodes={tname}=nodes_{tname}.csv")
    for rel, df in edges.groupby("rel"):
        f = df[["src_id", "dst_id", "rel", "weight", "source", "props"]]
        f.to_csv(out_dir / f"edges_{rel}.csv", index=False,
                 header=[":START_ID", ":END_ID", ":TYPE", "weight:double", "source", "props"])
        args.append(f"--relationships=edges_{rel}.csv")
    script = ("#!/usr/bin/env bash\n# Bulk-load into an empty Neo4j 5 database (stop the DB first).\n"
              "set -euo pipefail\ncd \"$(dirname \"$0\")\"\nDB=${1:-neo4j}\n"
              "neo4j-admin database import full --overwrite-destination --array-delimiter=';' "
              "--multiline-fields=true \\\n  "
              + " \\\n  ".join(args) + " \\\n  \"$DB\"\n")
    (out_dir / "import.sh").write_text(script)
    (out_dir / "import.sh").chmod(0o755)


def run(cfg: Config) -> dict:
    nodes, edges = load_processed(cfg)
    schema.validate(nodes, edges).raise_if_failed()
    out = cfg.paths.exports
    ecfg = cfg.get("export", {})
    write_parquet(unified_nodes(nodes), out / "nodes.parquet")
    write_parquet(edges, out / "edges.parquet")
    if ecfg.get("graphml", True):
        g = to_networkx(nodes, edges, include_abstract=ecfg.get("graphml_include_abstract", False))
        nx.write_graphml(g, out / "graph.graphml")
    if ecfg.get("neo4j_csv", True):
        export_neo4j(nodes, edges, out / "neo4j")
    write_json(out / "schema.json", schema.schema_as_dict())
    manifest = {
        "run_name": cfg.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_file": str(cfg.path), "config": cfg.data,
        "counts": {"nodes": {k: len(v) for k, v in nodes.items()},
                   "edges": edges["rel"].value_counts().to_dict()},
        "reports": {name: read_json(cfg.paths.processed / f"{name}_report.json")
                    for name in ("collect", "select", "build")},
    }
    write_json(out / "manifest.json", manifest)
    log.info("export: wrote %s", out)
    return manifest["counts"]
