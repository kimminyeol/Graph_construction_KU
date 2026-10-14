"""LLM verification of entity-linking candidates (tier t3 of kg/linking.py).

Pairs are judged in batches of `batch` per request; every verdict is stored per pair in
data/raw/llm_link/pairs__<model>__<prompt>.jsonl, so reruns and other runs only pay for pairs never judged before.
"""
from __future__ import annotations

import hashlib
import json
import logging

from .config import Config
from .io import iter_jsonl
from .sources.llm import LLMClient

log = logging.getLogger(__name__)
PROMPT_VERSION = "v1"
PRICES = {"gpt-4.1": {"in": 2.00, "cached_in": 0.50, "out": 8.00}, "gpt-4o-mini": {"in": 0.15, "cached_in": 0.075, "out": 0.60}, "gpt-4.1-mini": {"in": 0.40, "cached_in": 0.10, "out": 1.60}}

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answers"],
          "properties": {"answers": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "same"],
                                                                 "properties": {"id": {"type": "integer"}, "same": {"type": "boolean"}}}}}}

_V1 = """You decide whether two names from a research-paper knowledge graph denote the SAME entity (a method, model, dataset, task or metric).
Answer "same": true only if the two names are interchangeable in a paper: spelling or hyphenation variants, plural vs singular, an acronym and its expansion, a trailing generic word ("test set" / "test dataset"), or a well-known alias.
Answer false when:
- one is broader or narrower than the other ("object detection" vs "few-shot object detection", "gradient descent" vs "stochastic gradient descent", "summarization" vs "extractive summarization");
- they are different versions, sizes or variants ("GPT-4" vs "GPT-4o", "Llama 2" vs "Llama 3", "BERT-base" vs "BERT-large");
- they are related but different things ("traffic sign detection" vs "traffic sign recognition", "adaptive boosting" vs "gradient boosting", "speech recognition" vs "speech synthesis");
- a family and one member of it ("YOLOv8" vs "YOLOv8n", "Mixtral" vs "Mixtral-8x7B");
- an acronym is ambiguous and you cannot be sure from the names and the optional context titles.
When unsure, answer false: a wrong merge is worse than a missed one.
Input: a JSON list of {"id", "type", "a", "b", optionally "ctx_a", "ctx_b" = a title of a paper that uses that name}. Output one answer per id."""


_V2 = _V1.replace("""When unsure, answer false: a wrong merge is worse than a missed one.""",
"""- one name has an extra content word that narrows it ("test set" vs "private test set", "summarization" vs "abstractive summarization", "LLM-based agents" vs "LLM-based system");
  only words that carry no meaning of their own are ignorable: dataset, data, model, method, approach, score, metric, task, "-based", "-level", and a vendor or organisation prefix ("MITRE ATT&CK" = "ATT&CK").
Examples of same: "LSTM" / "long short-term memory", "F1" / "F1-score", "knowledge graphs" / "knowledge graph", "GAN" / "generative adversarial network", "image classification task" / "image classification", "Word2Vec" / "word2vec".
Examples of different: "BERT" / "RoBERTa", "precision" / "precision@k", "CNN" / "RNN", "question answering" / "visual question answering", "accuracy" / "top-5 accuracy".
When unsure, answer false: a wrong merge is worse than a missed one.""")
PROMPTS = {"v1": _V1, "v2": _V2}


def pair_key(typ: str, a: str, b: str) -> str:
    x, y = sorted((a.strip().lower(), b.strip().lower()))
    return hashlib.sha1(f"{typ}|{x}|{y}".encode()).hexdigest()[:20]


def _stage(cfg: Config, lcfg: dict):
    """One judging stage: returns judge(pairs) -> list[bool]."""
    model = lcfg.get("model", "gpt-4o-mini")
    ver = lcfg.get("prompt_version", PROMPT_VERSION)
    base = cfg.paths.raw / "llm_link"
    pairs_path = base / f"pairs__{model}__{ver}.jsonl"
    store = {r["key"]: r["same"] for r in iter_jsonl(pairs_path)}
    client = LLMClient(model, base / f"batches__{model}__{ver}.jsonl", params=lcfg.get("params", {"temperature": 0, "max_tokens": 1500}),
                       price=lcfg.get("price") or PRICES.get(model), max_usd=lcfg.get("max_usd"), workers=lcfg.get("workers", 16))
    size = lcfg.get("batch", 25)

    def judge(pairs: list[dict]) -> list[bool]:
        keys = [pair_key(p["type"], p["a"], p["b"]) for p in pairs]
        todo = [i for i, k in enumerate(keys) if k not in store]
        log.info("link judge: %d pairs, %d new (%s)", len(pairs), len(todo), model)
        batches = [todo[s:s + size] for s in range(0, len(todo), size)]
        items, owner = [], {}
        for b in batches:
            payload = []
            for n, i in enumerate(b):
                p = pairs[i]
                row = {"id": n, "type": p["type"], "a": p["a"], "b": p["b"]}
                if p.get("ctx_a"):
                    row["ctx_a"] = p["ctx_a"][0]
                if p.get("ctx_b"):
                    row["ctx_b"] = p["ctx_b"][0]
                payload.append(row)
            user = json.dumps(payload, ensure_ascii=False)
            k = hashlib.sha1((ver + user).encode()).hexdigest()[:24]
            items.append((k, PROMPTS[ver], user))
            owner[k] = b
        res = client.run_many(items, SCHEMA)
        log.info("link judge: %s", {k: v for k, v in res.items() if k != "tokens"} | {"out_tokens": res["tokens"]["out"]})
        new = []
        for k, b in owner.items():
            rec = client.cache.get(k)
            if rec is None:
                continue
            ans = {a["id"]: a["same"] for a in rec["content"].get("answers", [])}
            for n, i in enumerate(b):
                if n in ans:
                    store[keys[i]] = bool(ans[n])
                    new.append({"key": keys[i], "same": bool(ans[n]), "a": pairs[i]["a"], "b": pairs[i]["b"], "type": pairs[i]["type"]})
        if new:
            pairs_path.parent.mkdir(parents=True, exist_ok=True)
            with pairs_path.open("a") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        missing = sum(k not in store for k in keys)
        if missing:
            log.warning("link judge: %d pairs without a verdict (treated as different); rerun to resume", missing)
        return [store.get(k, False) for k in keys]

    return judge


def make_judge(cfg: Config, lcfg: dict):
    """`link_judge.stages` = [{model, prompt_version}, ...]: a pair must be accepted by every stage; later (stronger, dearer) stages only see
    the pairs the earlier stages accepted. Without `stages` the config itself describes the single stage."""
    stages = [_stage(cfg, {**lcfg, **st}) for st in lcfg.get("stages", [{}])]

    def judge(pairs: list[dict]) -> list[bool]:
        keep = list(range(len(pairs)))
        for n, st in enumerate(stages):
            verdict = st([pairs[i] for i in keep])
            log.info("link judge stage %d: %d -> %d pairs", n + 1, len(keep), sum(verdict))
            keep = [i for i, v in zip(keep, verdict) if v]
        ok = set(keep)
        return [i in ok for i in range(len(pairs))]

    return judge
