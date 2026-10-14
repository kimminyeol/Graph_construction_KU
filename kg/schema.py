"""Domain-agnostic Research KG schema.

The schema is fixed up front so later phases (Hugging Face linking, LLM
extraction, experiments, communities) and later domains (biology, physics...)
only add *data*, never new node/edge types.

ID convention: every node id is ``<prefix>:<local id>``, e.g.
``paper:W4385245566``, ``author:A5023888391``, ``topic:T10181``,
``model:hf/meta-llama/Llama-3-8B``. The prefix is unique per node type, so the
type of any id can be recovered from the id alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class NodeType:
    name: str
    prefix: str
    table: str
    description: str
    properties: tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeType:
    name: str
    src: tuple[str, ...]
    dst: tuple[str, ...]
    description: str
    properties: tuple[str, ...] = ()
    same_type: bool = False  # src and dst must have the same node type


# Columns every node table has, before its type-specific properties.
NODE_BASE_COLUMNS = ("node_id", "node_type", "name", "source", "source_id")
# Columns every edge row has. Type-specific properties go into `props` (JSON).
EDGE_COLUMNS = ("src_id", "rel", "dst_id", "src_type", "dst_type", "weight", "source", "props")


NODE_TYPES: dict[str, NodeType] = {n.name: n for n in [
    # --- bibliographic backbone (OpenAlex) ---
    NodeType("Paper", "paper", "papers", "A scholarly work.",
             ("doi", "year", "publication_date", "work_type", "language", "abstract",
              "cited_by_count", "fwci", "referenced_works_count", "n_refs_internal",
              "is_oa", "oa_url", "venue_id", "primary_topic_id", "referenced_works",
              "corpus_units", "metadata_status", "alt_ids")),
    NodeType("Author", "author", "authors", "A person who authored papers.",
             ("orcid", "id_source", "n_papers", "alt_ids")),
    NodeType("Institution", "inst", "institutions", "A research organisation.",
             ("ror", "country_code", "inst_type", "n_papers")),
    NodeType("Venue", "venue", "venues", "A journal, conference, or repository.",
             ("venue_type", "issn_l", "n_papers")),
    NodeType("Topic", "topic", "topics",
             "A research area at one level of the domain > field > subfield > topic hierarchy.",
             ("level", "n_papers")),
    # --- research content (Hugging Face + LLM extraction; later phases) ---
    NodeType("Method", "method", "methods", "A technique/algorithm/framework.", ("aliases", "n_papers")),
    NodeType("Model", "model", "models", "A (usually pretrained) model artifact.",
             ("aliases", "n_papers", "url", "hf_downloads", "hf_likes", "pipeline_tag", "library", "license")),
    NodeType("Dataset", "dataset", "datasets", "A dataset or benchmark.",
             ("aliases", "n_papers", "url", "hf_downloads", "hf_likes", "license")),
    NodeType("Task", "task", "tasks", "A problem setting, e.g. multi-hop QA.", ("aliases", "n_papers")),
    NodeType("Metric", "metric", "metrics", "An evaluation metric, e.g. Recall@5.", ("aliases", "n_papers")),
    NodeType("Experiment", "exp", "experiments",
             "One reported (method/model, dataset, metric) result in a paper.",
             ("paper_id", "score", "setting")),
    NodeType("Community", "community", "communities",
             "A cluster of related papers (graph community detection).",
             ("level", "summary", "keywords", "n_papers")),
]}

PREFIX_TO_TYPE = {n.prefix: n.name for n in NODE_TYPES.values()}
assert len(PREFIX_TO_TYPE) == len(NODE_TYPES), "node prefixes must be unique"

_ENTITY = ("Method", "Model", "Dataset", "Task", "Metric")

EDGE_TYPES: dict[str, EdgeType] = {e.name: e for e in [
    EdgeType("AUTHORED", ("Author",), ("Paper",), "Author wrote paper.",
             ("author_position", "is_corresponding")),
    EdgeType("AFFILIATED_WITH", ("Author",), ("Institution",),
             "Author listed this affiliation on corpus papers.", ("first_year", "last_year")),
    EdgeType("CITES", ("Paper",), ("Paper",), "Paper references paper."),
    EdgeType("SIMILAR_TO", ("Paper",), ("Paper",),
             "Content similarity (title+abstract embeddings). Symmetric: stored once per pair, "
             "src < dst by id; weight = cosine similarity.", ("model", "rank")),
    EdgeType("PUBLISHED_AT", ("Paper",), ("Venue",), "Primary publication venue."),
    EdgeType("HAS_TOPIC", ("Paper",), ("Topic",), "Topic assignment.", ("is_primary",)),
    EdgeType("SUBTOPIC_OF", ("Topic",), ("Topic",), "Topic hierarchy (child -> parent)."),
    EdgeType("PROPOSES", ("Paper",), ("Method", "Model", "Dataset", "Metric"),
             "Paper introduces this entity."),
    EdgeType("USES_METHOD", ("Paper", "Experiment"), ("Method",), "Uses a method."),
    EdgeType("USES_MODEL", ("Paper", "Experiment"), ("Model",), "Uses a model."),
    EdgeType("USES_DATASET", ("Paper",), ("Dataset",), "Uses a dataset."),
    EdgeType("ADDRESSES", ("Paper", "Experiment", "Model", "Dataset"), ("Task",),
             "Targets / is intended for a task."),
    EdgeType("REPORTS", ("Paper",), ("Experiment",), "Paper reports an experiment."),
    EdgeType("EVALUATED_ON", ("Experiment",), ("Dataset",), "Experiment's evaluation data."),
    EdgeType("USES_METRIC", ("Paper", "Experiment"), ("Metric",), "Metric a paper / an experiment reports."),
    EdgeType("TRAINED_ON", ("Model",), ("Dataset",), "Model training data (e.g. HF card)."),
    EdgeType("DERIVED_FROM", ("Model",), ("Model",), "Fine-tune / quantisation of a base model."),
    EdgeType("BELONGS_TO", ("Paper",), ("Community",), "Community membership."),
    EdgeType("SYNONYM_OF", _ENTITY, _ENTITY,
             "Alias -> canonical entity. Kept as an edge (not merged) to preserve provenance.",
             ("similarity", "method"), same_type=True),
]}


def make_id(node_type: str, local_id: str) -> str:
    return f"{NODE_TYPES[node_type].prefix}:{local_id}"


def type_of(node_id: str) -> str:
    return PREFIX_TO_TYPE[node_id.split(":", 1)[0]]


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError("KG validation failed:\n  " + "\n  ".join(self.errors))


def validate(nodes: dict[str, pd.DataFrame], edges: pd.DataFrame) -> ValidationReport:
    """Check node tables and edges against the schema.

    ``nodes`` maps node type name -> table. Checks: known types, id prefixes,
    unique ids, required columns, edge types / endpoint types, dangling edges.
    """
    rep = ValidationReport()
    all_ids: set[str] = set()
    for tname, df in nodes.items():
        if tname not in NODE_TYPES:
            rep.errors.append(f"unknown node type {tname!r}")
            continue
        missing = set(NODE_BASE_COLUMNS) - set(df.columns)
        if missing:
            rep.errors.append(f"{tname}: missing columns {sorted(missing)}")
            continue
        prefix = NODE_TYPES[tname].prefix + ":"
        bad = df.loc[~df["node_id"].str.startswith(prefix), "node_id"]
        if len(bad):
            rep.errors.append(f"{tname}: {len(bad)} ids without prefix {prefix!r}, e.g. {bad.iloc[0]}")
        dup = df["node_id"].duplicated()
        if dup.any():
            rep.errors.append(f"{tname}: {int(dup.sum())} duplicate ids, e.g. {df.loc[dup, 'node_id'].iloc[0]}")
        if (df["node_type"] != tname).any():
            rep.errors.append(f"{tname}: rows with wrong node_type")
        all_ids.update(df["node_id"])

    if edges.empty:
        return rep
    missing = set(EDGE_COLUMNS) - set(edges.columns)
    if missing:
        rep.errors.append(f"edges: missing columns {sorted(missing)}")
        return rep
    for (rel, st, dt), n in edges.groupby(["rel", "src_type", "dst_type"]).size().items():
        et = EDGE_TYPES.get(rel)
        if et is None:
            rep.errors.append(f"unknown edge type {rel!r} ({n} edges)")
        elif st not in et.src or dt not in et.dst or (et.same_type and st != dt):
            rep.errors.append(f"{rel}: {st}->{dt} not allowed by schema ({n} edges)")
    dangling = ~edges["src_id"].isin(all_ids) | ~edges["dst_id"].isin(all_ids)
    if dangling.any():
        row = edges.loc[dangling].iloc[0]
        rep.errors.append(f"{int(dangling.sum())} dangling edges, e.g. {row.src_id} -{row.rel}-> {row.dst_id}")
    dup = edges.duplicated(["src_id", "rel", "dst_id"])
    if dup.any():
        rep.errors.append(f"{int(dup.sum())} duplicate edges")
    return rep


def schema_as_dict() -> dict:
    return {
        "id_convention": "<prefix>:<local id>",
        "node_base_columns": list(NODE_BASE_COLUMNS),
        "edge_columns": list(EDGE_COLUMNS),
        "node_types": {n.name: {"prefix": n.prefix, "table": n.table, "description": n.description,
                                "properties": list(n.properties)} for n in NODE_TYPES.values()},
        "edge_types": {e.name: {"src": list(e.src), "dst": list(e.dst), "description": e.description,
                                "properties": list(e.properties), "same_type": e.same_type}
                       for e in EDGE_TYPES.values()},
    }
