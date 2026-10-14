"""Abstract -> research entities: prompt, JSON schema and the rule-based post-processing.

Shared by the `extract` / `content` steps and by the evaluation, so the scored output is exactly what enters the graph.
Post-processing is deliberately rule-based and cheap: it is the precision guard around the LLM.

  grounding   an entity name must appear in the title/abstract (token-aligned, hyphen/space/case-insensitive);
              a name the model produced from its own knowledge is dropped
  generic     "deep learning", "large language models", ... are not entities
  dedupe      one entry per (type, normalised name); `proposed` wins over `used`
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

PROMPT_VERSION = "v2"
TYPES = ("task", "method", "model", "dataset", "metric")
ROLES = ("proposed", "used")

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["entities", "results"],
    "properties": {
        "entities": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "type", "role"],
            "properties": {"name": {"type": "string"}, "type": {"type": "string", "enum": list(TYPES)},
                           "role": {"type": "string", "enum": list(ROLES)}}}},
        "results": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["subject", "dataset", "metric", "score"],
            "properties": {k: {"type": "string"} for k in ("subject", "dataset", "metric", "score")}}},
    },
}

_SYSTEM_V1 = """You extract research entities from the title and abstract of one AI / computer-science paper.
Use ONLY what the text states explicitly. Never add an entity from your own background knowledge.

Entity types
- task: the problem or application the paper addresses, e.g. "question answering", "image segmentation". The 1-4 most central.
- method: a named or clearly distinct technique, algorithm, framework, training recipe or architecture, e.g. "LoRA", "chain-of-thought prompting".
- model: a specific named model or model family (pretrained or trained in the paper), e.g. "GPT-4", "Llama 2", "ResNet-50".
- dataset: a named dataset or benchmark, e.g. "MMLU", "ImageNet".
- metric: a named evaluation measure, e.g. "F1", "Recall@5", "BLEU".

role
- "proposed": the paper introduces this method / model / dataset / benchmark / metric (new in this paper).
- "used": it already exists and the paper uses, builds on, compares with or evaluates on it. Tasks and metrics are always "used".

Rules
- name: as written in the text. For "Full Name (ACRONYM)" use the ACRONYM when it is the common name of a well-known model, dataset or metric, otherwise the full name. No parentheses, no trailing words like "model" or "dataset" unless part of the name.
- Skip generic terms: "deep learning", "machine learning", "neural networks", "large language models", "LLMs", "AI", "transformers".
- Skip an unnamed "our method" / "the proposed approach"; list only things that have a name or a distinct technical description.
- At most 5 tasks, 8 methods, 8 models, 8 datasets, 6 metrics. Use empty arrays when nothing qualifies.
- results: only when the text gives a number for a metric or dataset. subject = the method/model that achieves it, as named in the text ("" if unnamed); dataset and metric = as named in the text ("" if not stated); score = the number as written (e.g. "92.3%", "+4.1 points"). At most 5.

Example 1
Title: FastRAG: Efficient Retrieval-Augmented Generation with Adaptive Chunking
Abstract: Retrieval-augmented generation (RAG) improves the factuality of large language models, but fixed-size chunking hurts retrieval quality. We propose FastRAG, a retrieval framework that adapts chunk boundaries to document structure. On Natural Questions and HotpotQA, FastRAG built on Llama 3 8B improves Recall@5 from 61.2% to 68.9% over a BM25 baseline while cutting indexing time by 40%. We also release ChunkBench, a benchmark of 2,000 annotated documents.
Output: {"entities":[{"name":"retrieval","type":"task","role":"used"},{"name":"FastRAG","type":"method","role":"proposed"},{"name":"retrieval-augmented generation","type":"method","role":"used"},{"name":"BM25","type":"method","role":"used"},{"name":"Llama 3 8B","type":"model","role":"used"},{"name":"Natural Questions","type":"dataset","role":"used"},{"name":"HotpotQA","type":"dataset","role":"used"},{"name":"ChunkBench","type":"dataset","role":"proposed"},{"name":"Recall@5","type":"metric","role":"used"}],"results":[{"subject":"FastRAG","dataset":"","metric":"Recall@5","score":"68.9%"}]}

Example 2
Title: A Survey of Graph Neural Networks for Drug Discovery
Abstract: Graph neural networks (GNNs) have become a standard tool for learning on molecules. We review molecular property prediction, drug-target interaction and de novo molecular generation, summarise widely used benchmarks such as MoleculeNet, and discuss open problems including data scarcity.
Output: {"entities":[{"name":"molecular property prediction","type":"task","role":"used"},{"name":"drug-target interaction","type":"task","role":"used"},{"name":"de novo molecular generation","type":"task","role":"used"},{"name":"graph neural networks","type":"method","role":"used"},{"name":"MoleculeNet","type":"dataset","role":"used"}],"results":[]}"""

_SYSTEM_V2 = _SYSTEM_V1.replace(
"""- task: the problem or application the paper addresses, e.g. "question answering", "image segmentation". The 1-4 most central.
- method: a named or clearly distinct technique, algorithm, framework, training recipe or architecture, e.g. "LoRA", "chain-of-thought prompting".
- model: a specific named model or model family (pretrained or trained in the paper), e.g. "GPT-4", "Llama 2", "ResNet-50".
- dataset: a named dataset or benchmark, e.g. "MMLU", "ImageNet".
- metric: a named evaluation measure, e.g. "F1", "Recall@5", "BLEU".""",
"""- task: a concrete problem or application the paper addresses, e.g. "question answering", "image segmentation". The 1-4 most central. Not a broad field, learning paradigm or model class ("reinforcement learning", "large language models", "deep learning").
- method: a named or clearly distinct technique, algorithm, framework or module that is not itself a trained model, e.g. "LoRA", "chain-of-thought prompting", "Belady's algorithm".
- model: a specific named model, model family or named network architecture that is trained, fine-tuned or evaluated, e.g. "GPT-4", "Llama 2", "ResNet-50", "DenseNet", "BERT". A paper's own named system or network (e.g. "MPSA-DenseNet") is a model.
- dataset: a named dataset, benchmark or named data source, e.g. "MMLU", "ImageNet".
- metric: a named evaluation measure that the paper uses or reports, e.g. "F1", "Recall@5", "BLEU", "accuracy", "WER".""").replace(
"""- Skip generic terms: "deep learning", "machine learning", "neural networks", "large language models", "LLMs", "AI", "transformers".
- Skip an unnamed "our method" / "the proposed approach"; list only things that have a name or a distinct technical description.""",
"""- If the text defines an acronym ("word error rate (WER)"), use the acronym.
- Skip generic terms ("deep learning", "machine learning", "neural networks", "large language models", "LLMs", "AI", "transformers"), classes of methods ("supervised methods", "traditional methods", "hybrid approaches"), generic systems or applications, and unnamed components or modules of the paper's own framework.
- Skip an unnamed "our method" / "the proposed approach"; list only things that have a name or a distinct technical description.""")

PROMPTS = {"v1": _SYSTEM_V1, "v2": _SYSTEM_V2}


def system_prompt(version: str = PROMPT_VERSION) -> str:
    return PROMPTS[version]


def user_message(title: str, abstract: str) -> str:
    return f"Title: {title}\nAbstract: {abstract}"


def cache_key(title: str, abstract: str) -> str:
    return hashlib.sha256(f"{title}\n{abstract}".encode()).hexdigest()[:24]


# --- normalisation ------------------------------------------------------------------------------------------------------
def entity_key(name: str) -> str:
    """Case, spacing, hyphen and parenthetical-insensitive key: "GPT-4" == "gpt 4" == "GPT4"."""
    s = unicodedata.normalize("NFKC", name).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    return re.sub(r"[^\w+#@]|_", "", s)[:80]


def text_tokens(text: str) -> list[str]:
    return re.findall(r"[\w+#@]+", unicodedata.normalize("NFKC", text).lower())


def grounded(name: str, tokens: list[str]) -> bool:
    """Does `name` occur in the token list, aligned on token boundaries (so "gpt4" ~ "GPT-4" but "bert" !~ "robert")?"""
    key = entity_key(name)
    if not key:
        return False
    variants = {key, key + "s", key[:-1] if key.endswith("s") else key, key + "es"}
    for i in range(len(tokens)):
        acc = ""
        for j in range(i, min(i + 12, len(tokens))):
            acc += tokens[j]
            if acc in variants:
                return True
            if len(acc) > len(key) + 2:
                break
    return False



# --- abbreviations ("word error rate (WER)") ----------------------------------------------------------------------------
def _long_form(lf: str, sf: str) -> str | None:
    """Schwartz-Hearst: the shortest suffix of `lf` that spells out `sf`."""
    s, l = len(sf) - 1, len(lf) - 1
    while s >= 0:
        c = sf[s].lower()
        if not c.isalnum():
            s -= 1
            continue
        while l >= 0 and (lf[l].lower() != c or (s == 0 and l > 0 and lf[l - 1].isalnum())):
            l -= 1
        if l < 0:
            return None
        l -= 1
        s -= 1
    return lf[lf.rfind(" ", 0, l + 1) + 1:]


def abbreviation_pairs(text: str) -> list[tuple[str, str]]:
    """[(long form, short form)] for every "Long Form (SF)" in the text."""
    out = []
    for m in re.finditer(r"\(([^()]{2,20})\)", text):
        sf = m.group(1).strip()
        n_alnum = sum(c.isalnum() for c in sf)
        if len(sf.split()) > 2 or n_alnum < 2 or not sf[0].isalnum() or not any(c.isalpha() for c in sf) or sf.islower() and len(sf) > 6:
            continue
        words = text[:m.start()].split()
        lf = " ".join(words[-min(n_alnum + 5, n_alnum * 2):])
        lf = _long_form(lf, sf)
        if lf and len(lf.split()) >= 2 and len(lf) > len(sf) and entity_key(lf) != entity_key(sf):
            out.append((lf, sf))
    return out


def abbreviations(text: str) -> dict[str, tuple[str, str]]:
    """{key(long form): (key(short form), short form)} for every "Long Form (SF)" in the text."""
    out = {}
    for lf, sf in abbreviation_pairs(text):
        out.setdefault(entity_key(lf), (entity_key(sf), sf))
    return out


def canonical(name: str, aliases: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """(key, surface) with a long form replaced by the acronym the paper itself defines."""
    key = entity_key(name)
    return aliases[key] if key in aliases else (key, name)


GENERIC = {entity_key(x) for x in """deep learning|machine learning|neural network|neural networks|artificial intelligence|ai|llm|llms
large language model|large language models|language model|language models|foundation model|foundation models|generative ai
transformer|transformers|deep neural network|deep neural networks|pretrained model|pretrained models|pre-trained model
generative model|generative models|ai model|ai models|ml|nlp|natural language processing|computer vision|big data|data
generative artificial intelligence|genai|pre-trained language models|pretrained language models|plms|vision-language models""".replace("\n", "|").split("|")}
# learning paradigms are methods, but never a task
TASK_GENERIC = GENERIC | {entity_key(x) for x in "reinforcement learning|supervised learning|unsupervised learning|semi-supervised learning|self-supervised learning|transfer learning|federated learning|graph learning|representation learning".split("|")}
# "supervised methods", "traditional approaches", "mobile application": a class, not a named thing
CLASS_NAME = re.compile(r"^(\S+\s+){0,2}(methods?|approaches|techniques|methodologies|systems|applications)$", re.I)


def postprocess(content: dict, title: str, abstract: str, ground: tuple[str, ...] = ("method", "model", "dataset", "metric")) -> dict:
    """Raw model output -> {"entities": [...], "results": [...], "dropped": {reason: n}}."""
    toks = text_tokens(f"{title} {abstract}")
    digits = re.sub(r"[^0-9.]", "", f"{title} {abstract}")
    aliases = abbreviations(f"{title}. {abstract}")
    dropped = {"empty": 0, "generic": 0, "ungrounded": 0, "duplicate": 0}
    seen: dict[tuple[str, str], dict] = {}
    for e in content.get("entities", []):
        name, typ, role = (e.get("name") or "").strip(), e.get("type"), e.get("role")
        key = entity_key(name)
        if typ not in TYPES or not key or len(name) > 100:
            dropped["empty"] += 1
            continue
        if key in (TASK_GENERIC if typ == "task" else GENERIC) or (typ == "method" and CLASS_NAME.match(name)):
            dropped["generic"] += 1
            continue
        if typ in ground and not grounded(name, toks):
            dropped["ungrounded"] += 1
            continue
        key, name = canonical(name, aliases)
        role = "used" if typ in ("task", "metric") or role not in ROLES else role
        if (typ, key) in seen:
            dropped["duplicate"] += 1
            if role == "proposed":
                seen[(typ, key)]["role"] = "proposed"
            continue
        seen[(typ, key)] = {"type": typ, "name": name, "key": key, "role": role}
    results = []
    for r in content.get("results", []):
        score = (r.get("score") or "").strip()
        num = re.sub(r"[^0-9.]", "", score).strip(".")
        ds, mt, sj = ((r.get(k) or "").strip() for k in ("dataset", "metric", "subject"))
        if not score or not (ds or mt) or (num and num not in digits):
            continue
        results.append({"subject": sj if sj and grounded(sj, toks) else "", "dataset": ds if grounded(ds, toks) else "",
                        "metric": mt if grounded(mt, toks) else "", "score": score})
    results = [r for r in results if r["dataset"] or r["metric"]][:5]
    return {"entities": list(seen.values()), "results": results, "dropped": dropped}
