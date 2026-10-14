"""Pipeline runner.

  python -m kg.pipeline                                  # build the graph (every step after `collect`)
  python -m kg.pipeline --steps link similar communities  # selected steps
  python -m kg.pipeline --from extract                    # from a step to the end
  python -m kg.pipeline --list-steps

Each step only reads files written by earlier steps, so any step can be re-run
on its own. A new step is a module with `run(cfg) -> dict`, registered in STEPS.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Callable

from .config import Config, load_config
from .io import write_json
from .steps import (arxiv_refs, build_base_graph, collect_openalex, communities, content_graph, export_graph, extract_entities, graph_stats,
                    link_entities, s2_references, select_corpus, similar_papers, summarize_communities, verify_abstracts, verify_metadata)

STEPS: dict[str, tuple[Callable[[Config], dict], str]] = {
    "collect": (collect_openalex.run, "fetch candidate works from OpenAlex (resumable, shared raw store)"),
    "verify": (verify_metadata.run, "check records against arXiv / Crossref / DataCite; repair or drop corrupt ones"),
    "abstracts": (verify_abstracts.run, "repair or drop abstracts that belong to another work"),
    "select": (select_corpus.run, "pick the corpus and merge preprint / published versions"),
    "build": (build_base_graph.run, "Paper / Author / Institution / Venue / Topic nodes and edges"),
    "s2_refs": (s2_references.run, "Semantic Scholar reference lists -> extra CITES edges"),
    "arxiv_refs": (arxiv_refs.run, "CITES from arXiv HTML reference lists (rule-based)"),
    "extract": (extract_entities.run, "LLM extraction of concepts from abstracts (cached; costs money only for new abstracts)"),
    "content": (content_graph.run, "extractions -> Task / Method / Model / Dataset / Metric / Experiment nodes and edges"),
    "link": (link_entities.run, "SYNONYM_OF: normalisation, corpus acronym dictionary, embedding candidates + LLM check"),
    "similar": (similar_papers.run, "title + abstract embeddings -> SIMILAR_TO (GPU)"),
    "communities": (communities.run, "Leiden communities on the paper projection -> BELONGS_TO"),
    "summarize": (summarize_communities.run, "LLM name and summary per community (cached)"),
    "export": (export_graph.run, "nodes / edges parquet, GraphML, Neo4j CSV, schema"),
    "stats": (graph_stats.run, "validation and graph statistics"),
}
# the default run: everything after `collect` (the raw OpenAlex records are in data/raw)
BUILD = [s for s in STEPS if s != "collect"]


def _setup_logging(cfg: Config) -> None:
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    logfile = cfg.paths.logs / f"{cfg.run_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/kg.yaml")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--steps", nargs="+", choices=list(STEPS))
    g.add_argument("--from", dest="from_step", choices=list(STEPS))
    ap.add_argument("--list-steps", action="store_true")
    args = ap.parse_args(argv)

    if args.list_steps:
        for name, (_, desc) in STEPS.items():
            print(f"{name:10s} {desc}")
        return 0

    cfg = load_config(args.config)
    _setup_logging(cfg)
    # `kg.sh stop` sends SIGTERM: turn it into SystemExit so the status file records the stop.
    # Collection state is saved after every page, so the next run resumes from there.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("stopped by SIGTERM"))
    # Closing VSCode / the terminal / SSH sends SIGHUP: ignore it so even a directly started run survives
    # (scripts/kg.sh additionally detaches the process into its own session).
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    log = logging.getLogger("kg.pipeline")
    names = list(STEPS)
    todo = args.steps or (names[names.index(args.from_step):] if args.from_step else BUILD)
    todo = [n for n in names if n in todo]  # always run in pipeline order
    log.info("run=%s config=%s steps=%s", cfg.run_name, cfg.path.name, todo)
    # status file read by `scripts/kg.sh status` (works for detached runs too)
    status_path = cfg.paths.logs / f"{cfg.run_name}.status.json"
    status = {"run_name": cfg.run_name, "pid": os.getpid(), "config": str(cfg.path), "steps": todo,
              "started_at": datetime.now().isoformat(timespec="seconds"), "state": "running",
              "current_step": None, "done_steps": {}}
    write_json(status_path, status)
    for name in todo:
        t0 = time.time()
        log.info("=== %s ===", name)
        write_json(status_path, {**status, "current_step": name})
        try:
            result = STEPS[name][0](cfg)
        except BaseException as e:  # incl. KeyboardInterrupt / SIGTERM -> SystemExit
            stopped = isinstance(e, (SystemExit, KeyboardInterrupt))
            write_json(status_path, {**status, "state": "stopped" if stopped else "failed", "current_step": name,
                                     "error": f"{type(e).__name__}: {e}"[:1000],
                                     "ended_at": datetime.now().isoformat(timespec="seconds")})
            log.exception("step %s failed", name)
            raise
        status["done_steps"][name] = round(time.time() - t0, 1)
        log.info("=== %s done in %.1fs ===", name, time.time() - t0)
        if name == "collect" and not result.get("complete", True):
            log.warning("collection incomplete (%s); continuing with what was fetched", result.get("stopped_reason"))
            status["warning"] = f"collect incomplete: {result.get('stopped_reason')}"
    write_json(status_path, {**status, "state": "done", "current_step": None,
                             "ended_at": datetime.now().isoformat(timespec="seconds")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
