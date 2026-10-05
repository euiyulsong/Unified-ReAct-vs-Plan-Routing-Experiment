#!/usr/bin/env python3
"""
unified_dependency_router_eval.py

Unified experiment for the hypothesis:

    0 = ReAct
        Use when the NEXT action/query depends on intermediate observations.

    1 = Plan / Plan-then-Execute
        Use when the required action/search/tool set can be inferred upfront
        from the original request, including simple one-step tasks.

Datasets
--------
Retrieval:
  - SQuAD validation: 50
      dependency label = 1 (upfront/simple retrieval is sufficient)
  - HotpotQA distractor validation: 50
      dependency label = 0 (later retrieval often depends on earlier evidence)

Tool selection:
  - BFCL v3 multiple: 50 single-tool cases
      dependency label = 1 (required tool can be inferred upfront)
  - BFCL v3 parallel_multiple: 50 multi-tool cases
      dependency label = 1 (required tool set can be inferred upfront)

Methods
-------
  always_react
  always_plan
  router
  static_dependency   # diagnostic policy based on the hypothesis labels

IMPORTANT FAIRNESS RULE
-----------------------
Within each task family, the SAME executor functions are used by:
  - fixed baselines
  - router-selected execution
  - static_dependency

Router ONLY adds:
  question/task + environment description -> "0" or "1"

No JSON output from the LLM.

LLM outputs
-----------
Router:
    0
or:
    1

Retrieval ReAct follow-up:
    -
or:
    one follow-up search query

Retrieval Plan:
    one search query per line, max 3

BFCL ReAct:
    exact_tool_name
or:
    -

BFCL Plan:
    exact_tool_name
    exact_tool_name
    ...

Common end-to-end metrics
-------------------------
task_success:
  - SQuAD / HotpotQA: full supporting-document retrieval
  - BFCL: exact tool-set match

task_score:
  - Retrieval: support_recall
  - BFCL: tool_f1

Also prints:
  route_accuracy
  LLM calls
  cache hits
  latency
  tokens
  per-group results
  router confusion
  paired comparisons vs fixed baselines
  per-example oracle best of ReAct/Plan

Install
-------
pip install -U openai datasets rank-bm25 numpy pandas tqdm huggingface_hub

Run
---
export OPENAI_API_KEY="..."

python unified_dependency_router_eval.py \
  --n-squad 50 \
  --n-hotpot 50 \
  --n-bfcl-single 50 \
  --n-bfcl-multi 50 \
  --corpus-size 1000 \
  --top-k 3 \
  --max-steps 3 \
  --output unified_dependency_router_results.csv

For standalone latency/cost:
  add --no-cache
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-6-luna")
SEED_DEFAULT = 42

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_SINGLE_DATA = "BFCL_v3_multiple.json"
BFCL_SINGLE_ANS = "possible_answer/BFCL_v3_multiple.json"
BFCL_MULTI_DATA = "BFCL_v3_parallel_multiple.json"
BFCL_MULTI_ANS = "possible_answer/BFCL_v3_parallel_multiple.json"

client = OpenAI()
LLM_CACHE: Dict[str, Tuple[str, dict]] = {}


# =============================================================================
# HELPERS
# =============================================================================

def clean_text(x) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def fresh_meta() -> dict:
    return {
        "llm_calls": 0,
        "llm_cache_hits": 0,
        "latency_s": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def add_meta(a: dict, b: dict) -> dict:
    out = dict(a)
    for k in fresh_meta().keys():
        out[k] = out.get(k, 0) + b.get(k, 0)
    return out


# =============================================================================
# LLM - PLAIN TEXT ONLY
# =============================================================================

def llm_text(
    *,
    kind: str,
    prompt: str,
    model: str,
    effort: str,
    max_output_tokens: int,
    use_cache: bool,
) -> Tuple[str, dict]:

    cache_key = f"{model}\n{effort}\n{kind}\n{prompt}"

    if use_cache and cache_key in LLM_CACHE:
        text, _ = LLM_CACHE[cache_key]
        meta = fresh_meta()
        meta["llm_cache_hits"] = 1
        print(f"      [LLM:{kind}:CACHE] {text!r}", flush=True)
        return text, meta

    t0 = time.perf_counter()

    response = client.responses.create(
        model=model,
        reasoning={"effort": effort},
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    latency = time.perf_counter() - t0
    text = response.output_text.strip()

    usage = getattr(response, "usage", None)

    meta = {
        "llm_calls": 1,
        "llm_cache_hits": 0,
        "latency_s": latency,
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
    }

    if use_cache:
        LLM_CACHE[cache_key] = (text, meta)

    print(
        f"      [LLM:{kind}] {text!r} "
        f"(lat={latency:.3f}s, in={meta['input_tokens']}, out={meta['output_tokens']})",
        flush=True,
    )

    return text, meta


# =============================================================================
# ROUTER PROMPT
# =============================================================================

def router_prompt(item: dict) -> str:
    return f"""Choose the execution strategy.

Output EXACTLY one character:
0 = ReAct
1 = Plan / Plan-then-Execute

Choose 0 when the NEXT action or search query depends on information that will
only become known after observing an intermediate search/tool result.

Choose 1 when the required action(s), search(es), or tool set can be identified
upfront from the original request before execution. This includes simple
one-step tasks.

Key distinction:
- 0: the action graph must be DISCOVERED during execution.
- 1: the action graph is already inferable BEFORE execution.

Important:
- Do not decide based only on whether the request is "complex".
- Do not decide based only on the number of available tools.
- Multi-step can still be 1 if all required actions are known upfront.
- A sequential dependency is 0 when later actions require entities/facts found
  only from earlier observations.
- Output no explanation.
- Output no JSON.

Task:
{item["question"]}

Environment:
{item["router_environment"]}

Output only 0 or 1."""


def parse_route(text: str) -> int:
    s = text.strip()

    if s == "0":
        return 0
    if s == "1":
        return 1

    m = re.search(r"(?<!\d)([01])(?!\d)", s)
    if m:
        print(f"      [WARN] non-exact route={s!r}; parsed={m.group(1)}")
        return int(m.group(1))

    print(f"      [WARN] route parse failed={s!r}; fallback=0")
    return 0


def call_router(item, model, effort, use_cache):
    raw, meta = llm_text(
        kind="ROUTER",
        prompt=router_prompt(item),
        model=model,
        effort=effort,
        max_output_tokens=8,
        use_cache=use_cache,
    )

    route = parse_route(raw)

    print(
        f"      [ROUTE] {route} ({'REACT' if route == 0 else 'PLAN'})",
        flush=True,
    )

    return route, meta


# =============================================================================
# RETRIEVAL DATA
# =============================================================================

def build_squad(n: int, corpus_size: int, seed: int):
    ds = load_dataset("rajpurkar/squad", split="validation")
    rng = random.Random(seed)

    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    chosen = []
    seen_contexts = set()

    for i in idxs:
        ctx = clean_text(ds[i]["context"])
        cid = stable_hash(ctx)
        if cid in seen_contexts:
            continue
        seen_contexts.add(cid)
        chosen.append(i)
        if len(chosen) >= n:
            break

    corpus = {}
    items = []

    for i in chosen:
        row = ds[i]
        ctx = clean_text(row["context"])
        doc_id = f"squad::{stable_hash(ctx)}"

        corpus[doc_id] = {
            "id": doc_id,
            "title": clean_text(row.get("title", "")),
            "text": ctx,
        }

        items.append({
            "id": f"squad::{row['id']}",
            "family": "retrieval",
            "group": "squad",
            "question": clean_text(row["question"]),
            "gold_ids": {doc_id},
            # Empirical/hypothesis label:
            # one-step retrieval is inferable upfront.
            "dependency_route": 1,
            "router_environment": (
                "You have a document search tool: search(query). "
                "You may search up to 3 times. "
                "The task is to retrieve the document(s) needed to answer the question."
            ),
        })

    distractors = list(range(len(ds)))
    rng.shuffle(distractors)

    for i in distractors:
        if len(corpus) >= corpus_size:
            break

        row = ds[i]
        ctx = clean_text(row["context"])
        doc_id = f"squad::{stable_hash(ctx)}"

        if doc_id not in corpus:
            corpus[doc_id] = {
                "id": doc_id,
                "title": clean_text(row.get("title", "")),
                "text": ctx,
            }

    return items, list(corpus.values())


def hotpot_docs(row):
    out = []

    titles = row["context"]["title"]
    sentence_groups = row["context"]["sentences"]

    for title, sents in zip(titles, sentence_groups):
        title = clean_text(title)
        out.append({
            "id": f"hotpot::{title.lower()}",
            "title": title,
            "text": clean_text(" ".join(sents)),
        })

    return out


def build_hotpot(n: int, corpus_size: int, seed: int):
    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )

    rng = random.Random(seed + 1)
    chosen = rng.sample(range(len(ds)), n)

    corpus = {}
    items = []

    for i in chosen:
        row = ds[i]

        for d in hotpot_docs(row):
            corpus.setdefault(d["id"], d)

        gold_titles = {
            clean_text(t).lower()
            for t in row["supporting_facts"]["title"]
        }

        gold_ids = {f"hotpot::{t}" for t in gold_titles}

        items.append({
            "id": f"hotpot::{row['id']}",
            "family": "retrieval",
            "group": "hotpotqa",
            "question": clean_text(row["question"]),
            "gold_ids": gold_ids,
            # Empirical/hypothesis label:
            # later retrieval often depends on entities from earlier evidence.
            "dependency_route": 0,
            "router_environment": (
                "You have a document search tool: search(query). "
                "You may search up to 3 times. "
                "The task may require retrieving multiple supporting documents. "
                "Later queries can use entities discovered from earlier search results."
            ),
        })

    all_idxs = list(range(len(ds)))
    rng.shuffle(all_idxs)

    for i in all_idxs:
        if len(corpus) >= corpus_size:
            break

        for d in hotpot_docs(ds[i]):
            corpus.setdefault(d["id"], d)
            if len(corpus) >= corpus_size:
                break

    corpus_ids = set(corpus.keys())

    for item in items:
        missing = item["gold_ids"] - corpus_ids
        if missing:
            raise RuntimeError(f"Hotpot gold missing for {item['id']}: {missing}")

    return items, list(corpus.values())


# =============================================================================
# RETRIEVER
# =============================================================================

class BM25Search:
    def __init__(self, docs):
        self.docs = docs

        tokenized = [
            tokenize(f"{d.get('title', '')} {d['text']}")
            for d in docs
        ]

        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, idx in enumerate(order, start=1):
            d = self.docs[int(idx)]
            results.append({
                **d,
                "score": float(scores[int(idx)]),
                "rank": rank,
            })

        return results


# =============================================================================
# RETRIEVAL PROMPTS
# =============================================================================

def retrieval_plan_prompt(question: str) -> str:
    return f"""Create the minimal retrieval plan for this question.

Output ONLY 1 to 3 search queries, one per line.
No bullets.
No numbering.
No explanation.
No JSON.

Rules:
- If one search is enough, output one query.
- If multiple searches can be determined upfront, output them.
- Do not invent an unknown intermediate entity.
- Keep queries concise.

Question:
{question}"""


def retrieval_followup_prompt(
    question: str,
    previous_queries: List[str],
    evidence_text: str,
) -> str:
    prev = "\n".join(previous_queries) if previous_queries else "(none)"

    return f"""Decide whether another retrieval search is needed.

If the current evidence is sufficient, output exactly:
-

Otherwise output ONLY one focused follow-up search query.

Rules:
- No explanation.
- No JSON.
- Do not answer the original question.
- Do not repeat a previous query.
- Use entities discovered in the evidence when useful.

Question:
{question}

Previous queries:
{prev}

Evidence:
{evidence_text}

Output only "-" or one follow-up query."""


def parse_lines(text: str, max_lines: int) -> List[str]:
    out = []

    for raw in text.splitlines():
        s = raw.strip()

        if not s or s == "-":
            continue

        s = re.sub(r"^\s*[-*•]\s*", "", s)
        s = re.sub(r"^\s*\d+[\.\)]\s*", "", s)
        s = s.strip()

        if s:
            out.append(s)

        if len(out) >= max_lines:
            break

    return out


def parse_followup(text: str):
    s = text.strip()

    if s == "-":
        return None

    for line in s.splitlines():
        q = line.strip()
        if not q:
            continue
        if q == "-":
            return None
        q = re.sub(r"^\s*[-*•]\s*", "", q)
        q = re.sub(r"^\s*\d+[\.\)]\s*", "", q)
        return q.strip() or None

    return None


def format_evidence(evidence: Dict[str, dict], max_docs=9, chars_per_doc=900):
    blocks = []

    for i, d in enumerate(list(evidence.values())[:max_docs], start=1):
        blocks.append(
            f"[D{i}] {d.get('title', '')}\n"
            f"{d['text'][:chars_per_doc]}"
        )

    return "\n\n".join(blocks) if blocks else "(none)"


# =============================================================================
# RETRIEVAL EXECUTORS
# =============================================================================

def execute_retrieval_react(
    item,
    searcher,
    top_k,
    max_steps,
    model,
    effort,
    use_cache,
):
    meta = fresh_meta()
    queries = []
    retrieved_ids = []
    evidence = {}

    current_query = item["question"]

    for step in range(1, max_steps + 1):
        print(f"      [SEARCH {step}/{max_steps}] {current_query}", flush=True)

        results = searcher.search(current_query, top_k)

        for d in results:
            hit = "GOLD" if d["id"] in item["gold_ids"] else "----"
            print(
                f"          #{d['rank']} [{hit}] {d.get('title', '')} "
                f"score={d['score']:.3f}",
                flush=True,
            )

        queries.append(current_query)

        for d in results:
            retrieved_ids.append(d["id"])
            evidence.setdefault(d["id"], d)

        if step >= max_steps:
            break

        raw, m = llm_text(
            kind="RETRIEVAL_FOLLOWUP",
            prompt=retrieval_followup_prompt(
                item["question"],
                queries,
                format_evidence(evidence),
            ),
            model=model,
            effort=effort,
            max_output_tokens=64,
            use_cache=use_cache,
        )

        meta = add_meta(meta, m)

        next_query = parse_followup(raw)

        if next_query is None:
            print("      [FOLLOWUP] -", flush=True)
            break

        print(f"      [FOLLOWUP] {next_query}", flush=True)
        current_query = next_query

    return {
        "retrieved_ids": retrieved_ids,
        "queries": queries,
        "meta": meta,
    }


def execute_retrieval_plan(
    item,
    searcher,
    top_k,
    max_steps,
    model,
    effort,
    use_cache,
):
    """
    Plan-first retrieval:
      - one plan call
      - execute planned queries in order, max 3
      - NO extra ReAct follow-up call

    This directly tests whether the search graph can be specified upfront.
    """
    meta = fresh_meta()

    raw, m = llm_text(
        kind="RETRIEVAL_PLAN",
        prompt=retrieval_plan_prompt(item["question"]),
        model=model,
        effort=effort,
        max_output_tokens=100,
        use_cache=use_cache,
    )

    meta = add_meta(meta, m)

    queries = parse_lines(raw, max_steps)

    if not queries:
        queries = [item["question"]]

    print(f"      [PLAN] {queries}", flush=True)

    retrieved_ids = []

    for step, query in enumerate(queries[:max_steps], start=1):
        print(f"      [SEARCH {step}/{max_steps}] {query}", flush=True)

        results = searcher.search(query, top_k)

        for d in results:
            hit = "GOLD" if d["id"] in item["gold_ids"] else "----"
            print(
                f"          #{d['rank']} [{hit}] {d.get('title', '')} "
                f"score={d['score']:.3f}",
                flush=True,
            )

        for d in results:
            retrieved_ids.append(d["id"])

    return {
        "retrieved_ids": retrieved_ids,
        "queries": queries[:max_steps],
        "meta": meta,
    }


def retrieval_metrics(item, run):
    unique_ids = list(dict.fromkeys(run["retrieved_ids"]))
    ret = set(unique_ids)
    gold = set(item["gold_ids"])

    hit = gold & ret

    support_recall = len(hit) / len(gold) if gold else 0.0
    full_support = float(gold.issubset(ret)) if gold else 0.0

    first_rank = None
    for i, doc_id in enumerate(unique_ids, start=1):
        if doc_id in gold:
            first_rank = i
            break

    mrr_first = 1.0 / first_rank if first_rank else 0.0

    return {
        "task_success": full_support,
        "task_score": support_recall,
        "support_recall": support_recall,
        "full_support": full_support,
        "mrr_first": mrr_first,
        "tool_exact_match": np.nan,
        "tool_f1": np.nan,
        "execution_steps": len(run["queries"]),
    }


# =============================================================================
# BFCL DATA
# =============================================================================

def bfcl_load_jsonl(filename: str):
    path = hf_hub_download(
        repo_id=BFCL_REPO,
        filename=filename,
        repo_type="dataset",
    )

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def bfcl_answer_map(filename: str):
    return {
        row["id"]: row
        for row in bfcl_load_jsonl(filename)
    }


def bfcl_question(row):
    q = row.get("question", "")

    if isinstance(q, str):
        return clean_text(q)

    texts = []

    def walk(x):
        if isinstance(x, dict):
            if x.get("role") == "user" and "content" in x:
                texts.append(str(x["content"]))
            else:
                for v in x.values():
                    walk(v)
        elif isinstance(x, list):
            for y in x:
                walk(y)

    walk(q)
    return clean_text(" ".join(texts))


def bfcl_gold_tools(ans):
    names = []

    for call in ans.get("ground_truth", []):
        if isinstance(call, dict):
            names.extend(call.keys())

    return list(dict.fromkeys(names))


def tool_names(functions):
    return [
        x.get("name", "").strip()
        for x in functions
        if x.get("name")
    ]


def tool_docs(functions):
    blocks = []

    for i, fn in enumerate(functions, start=1):
        name = clean_text(fn.get("name", ""))
        desc = clean_text(fn.get("description", ""))

        params = fn.get("parameters") or {}
        req = params.get("required") or []
        req_text = ", ".join(str(x) for x in req) if req else "(none)"

        blocks.append(
            f"[{i}] {name}\n"
            f"    {desc}\n"
            f"    required: {req_text}"
        )

    return "\n".join(blocks)


def build_bfcl(n_single: int, n_multi: int, seed: int):
    single_rows = bfcl_load_jsonl(BFCL_SINGLE_DATA)
    single_ans = bfcl_answer_map(BFCL_SINGLE_ANS)

    multi_rows = bfcl_load_jsonl(BFCL_MULTI_DATA)
    multi_ans = bfcl_answer_map(BFCL_MULTI_ANS)

    singles = []

    for row in single_rows:
        rid = row["id"]
        if rid not in single_ans:
            continue

        gold = bfcl_gold_tools(single_ans[rid])
        funcs = row.get("function") or []
        cands = tool_names(funcs)

        if len(gold) != 1 or len(cands) < 2:
            continue

        if not set(gold).issubset(set(cands)):
            continue

        q = bfcl_question(row)

        singles.append({
            "id": f"bfcl_single::{rid}",
            "family": "tool",
            "group": "bfcl_single",
            "question": q,
            "functions": funcs,
            "candidate_tools": cands,
            "gold_tools": gold,
            "dependency_route": 1,
            "router_environment": (
                "You have the following candidate tools. "
                "Select only the tool capabilities actually required by the request.\n"
                + tool_docs(funcs)
            ),
        })

    multis = []

    for row in multi_rows:
        rid = row["id"]
        if rid not in multi_ans:
            continue

        gold = bfcl_gold_tools(multi_ans[rid])
        funcs = row.get("function") or []
        cands = tool_names(funcs)

        if not (2 <= len(gold) <= 3):
            continue

        if len(cands) < 2:
            continue

        if not set(gold).issubset(set(cands)):
            continue

        q = bfcl_question(row)

        multis.append({
            "id": f"bfcl_multi::{rid}",
            "family": "tool",
            "group": "bfcl_multi",
            "question": q,
            "functions": funcs,
            "candidate_tools": cands,
            "gold_tools": gold,
            "dependency_route": 1,
            "router_environment": (
                "You have the following candidate tools. "
                "Select only the tool capabilities actually required by the request.\n"
                + tool_docs(funcs)
            ),
        })

    rng = random.Random(seed + 10)

    if len(singles) < n_single:
        raise RuntimeError(f"Not enough BFCL single: {len(singles)}")
    if len(multis) < n_multi:
        raise RuntimeError(f"Not enough BFCL multi: {len(multis)}")

    return (
        rng.sample(singles, n_single)
        + rng.sample(multis, n_multi)
    )


# =============================================================================
# BFCL PROMPTS / EXECUTORS
# =============================================================================

def bfcl_react_prompt(item, selected):
    selected_text = "\n".join(selected) if selected else "(none)"

    return f"""Select the NEXT tool needed for this user request.

Output exactly one of:
- one exact tool name from Available tools
- a single hyphen: -

Use "-" only when no additional DISTINCT tool is needed.

Rules:
- Select only ONE tool per step.
- Do not repeat a previously selected tool.
- Do not output arguments.
- Do not explain.
- Do not output JSON.

User request:
{item["question"]}

Available tools:
{tool_docs(item["functions"])}

Already selected tools:
{selected_text}

Output one exact tool name or "-"."""


def bfcl_plan_prompt(item):
    return f"""Identify the DISTINCT tools needed to complete this user request.

Output ONLY exact tool names from Available tools.
Use one tool name per line.
Output at most 3 lines.

If only one tool is needed, output one line.
If multiple distinct tools are needed, output all of them, one per line.

Rules:
- Do not output arguments.
- Do not number the lines.
- Do not use bullets.
- Do not explain.
- Do not output JSON.
- Do not include unnecessary tools.

User request:
{item["question"]}

Available tools:
{tool_docs(item["functions"])}

Output only the needed exact tool name(s), one per line."""


def parse_tool_or_stop(text, candidates):
    s = text.strip()

    if s == "-":
        return None

    if s in candidates:
        return s

    lines = [x.strip() for x in s.splitlines() if x.strip()]

    if lines and lines[0] in candidates:
        return lines[0]

    hits = [name for name in candidates if name in s]

    if len(hits) == 1:
        return hits[0]

    return None


def parse_plan_tools(text, candidates, max_tools):
    out = []

    for raw in text.splitlines():
        s = raw.strip()

        if not s or s == "-":
            continue

        s = re.sub(r"^\s*[-*•]\s*", "", s)
        s = re.sub(r"^\s*\d+[\.\)]\s*", "", s)
        s = s.strip()

        tool = None

        if s in candidates:
            tool = s
        else:
            hits = [name for name in candidates if name in s]
            if len(hits) == 1:
                tool = hits[0]

        if tool and tool not in out:
            out.append(tool)

        if len(out) >= max_tools:
            break

    return out


def execute_tool_react(item, max_steps, model, effort, use_cache):
    selected = []
    meta = fresh_meta()

    for step in range(1, max_steps + 1):
        print(f"      [REACT TOOL STEP {step}/{max_steps}] selected={selected}")

        raw, m = llm_text(
            kind="BFCL_REACT",
            prompt=bfcl_react_prompt(item, selected),
            model=model,
            effort=effort,
            max_output_tokens=40,
            use_cache=use_cache,
        )

        meta = add_meta(meta, m)

        tool = parse_tool_or_stop(raw, item["candidate_tools"])

        if tool is None:
            print("      [REACT TOOL] -")
            break

        print(f"      [REACT TOOL] {tool}")

        if tool in selected:
            break

        selected.append(tool)

    return {
        "selected_tools": selected,
        "meta": meta,
        "execution_steps": len(selected),
    }


def execute_tool_plan(item, max_steps, model, effort, use_cache):
    raw, meta = llm_text(
        kind="BFCL_PLAN",
        prompt=bfcl_plan_prompt(item),
        model=model,
        effort=effort,
        max_output_tokens=100,
        use_cache=use_cache,
    )

    tools = parse_plan_tools(
        raw,
        item["candidate_tools"],
        max_steps,
    )

    print(f"      [PLAN TOOLS] {tools}")

    return {
        "selected_tools": tools,
        "meta": meta,
        "execution_steps": len(tools),
    }


def tool_metrics_common(item, run):
    gold = set(item["gold_tools"])
    pred = set(run["selected_tools"])

    tp = len(gold & pred)

    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 1.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    em = float(pred == gold)

    return {
        "task_success": em,
        "task_score": f1,
        "support_recall": np.nan,
        "full_support": np.nan,
        "mrr_first": np.nan,
        "tool_exact_match": em,
        "tool_f1": f1,
        "execution_steps": run["execution_steps"],
    }


# =============================================================================
# UNIFIED METHOD EVALUATION
# =============================================================================

def evaluate_method(
    item,
    method,
    retrieval_searchers,
    top_k,
    max_steps,
    model,
    effort,
    use_cache,
):
    total_meta = fresh_meta()

    if method == "always_react":
        route = 0

    elif method == "always_plan":
        route = 1

    elif method == "router":
        route, m = call_router(
            item,
            model,
            effort,
            use_cache,
        )
        total_meta = add_meta(total_meta, m)

    elif method == "static_dependency":
        route = int(item["dependency_route"])
        print(
            f"      [STATIC DEPENDENCY ROUTE] {route} "
            f"({'REACT' if route == 0 else 'PLAN'})"
        )

    else:
        raise ValueError(method)

    if item["family"] == "retrieval":
        searcher = retrieval_searchers[item["group"]]

        if route == 0:
            run = execute_retrieval_react(
                item,
                searcher,
                top_k,
                max_steps,
                model,
                effort,
                use_cache,
            )
        else:
            run = execute_retrieval_plan(
                item,
                searcher,
                top_k,
                max_steps,
                model,
                effort,
                use_cache,
            )

        metrics = retrieval_metrics(item, run)

    elif item["family"] == "tool":
        if route == 0:
            run = execute_tool_react(
                item,
                max_steps,
                model,
                effort,
                use_cache,
            )
        else:
            run = execute_tool_plan(
                item,
                max_steps,
                model,
                effort,
                use_cache,
            )

        metrics = tool_metrics_common(item, run)

    else:
        raise ValueError(item["family"])

    total_meta = add_meta(total_meta, run["meta"])

    route_acc = float(route == item["dependency_route"])

    print(
        "      [RESULT] "
        f"route={'REACT' if route == 0 else 'PLAN'} "
        f"route_acc={route_acc:.0f} "
        f"task_success={metrics['task_success']:.3f} "
        f"task_score={metrics['task_score']:.3f} "
        f"steps={metrics['execution_steps']} "
        f"llm_calls={total_meta['llm_calls']}",
        flush=True,
    )

    return {
        "id": item["id"],
        "family": item["family"],
        "group": item["group"],
        "question": item["question"],
        "method": method,
        "gold_route": item["dependency_route"],
        "pred_route": route,
        "route_name": "REACT" if route == 0 else "PLAN",
        "route_accuracy": route_acc,
        **metrics,
        "llm_calls": total_meta["llm_calls"],
        "llm_cache_hits": total_meta["llm_cache_hits"],
        "latency_s": total_meta["latency_s"],
        "input_tokens": total_meta["input_tokens"],
        "output_tokens": total_meta["output_tokens"],
    }


# =============================================================================
# SUMMARY
# =============================================================================

SUMMARY_COLS = [
    "route_accuracy",
    "task_success",
    "task_score",
    "execution_steps",
    "llm_calls",
    "llm_cache_hits",
    "latency_s",
    "input_tokens",
    "output_tokens",
]


def print_summary(df):
    print("\n" + "=" * 100)
    print("OVERALL")
    print("=" * 100)

    overall = (
        df.groupby("method")[SUMMARY_COLS]
        .mean()
        .reset_index()
    )
    print(overall.to_string(index=False))

    print("\n" + "=" * 100)
    print("BY GROUP")
    print("=" * 100)

    by_group = (
        df.groupby(["group", "method"])[SUMMARY_COLS]
        .mean()
        .reset_index()
    )
    print(by_group.to_string(index=False))

    print("\n" + "=" * 100)
    print("BY FAMILY")
    print("=" * 100)

    by_family = (
        df.groupby(["family", "method"])[SUMMARY_COLS]
        .mean()
        .reset_index()
    )
    print(by_family.to_string(index=False))

    router_df = df[df["method"] == "router"].copy()

    if len(router_df):
        print("\n" + "=" * 100)
        print("ROUTER CONFUSION")
        print("=" * 100)

        cm = pd.crosstab(
            router_df["gold_route"],
            router_df["pred_route"],
            rownames=["gold"],
            colnames=["pred"],
            dropna=False,
        )
        print(cm.to_string())

        print("\n" + "=" * 100)
        print("ROUTER MIX")
        print("=" * 100)

        mix = (
            router_df.groupby(["group", "route_name"])
            .size()
            .rename("n")
            .reset_index()
        )

        mix["ratio"] = (
            mix.groupby("group")["n"]
            .transform(lambda s: s / s.sum())
        )

        print(mix.to_string(index=False))

    # Paired end-to-end comparison
    print("\n" + "=" * 100)
    print("PAIRED TASK-SUCCESS COMPARISON")
    print("=" * 100)

    pivot = df.pivot_table(
        index=["id", "group"],
        columns="method",
        values="task_success",
        aggfunc="first",
    )

    required = {"always_react", "always_plan", "router"}

    if required.issubset(set(pivot.columns)):
        for baseline in ["always_react", "always_plan"]:
            wins = (pivot["router"] > pivot[baseline]).sum()
            losses = (pivot["router"] < pivot[baseline]).sum()
            ties = (pivot["router"] == pivot[baseline]).sum()

            print(
                f"Router vs {baseline}: "
                f"win={wins}, loss={losses}, tie={ties}"
            )

        pivot["oracle_best"] = pivot[
            ["always_react", "always_plan"]
        ].max(axis=1)

        print()
        print(f"Always ReAct success = {pivot['always_react'].mean():.4f}")
        print(f"Always Plan success  = {pivot['always_plan'].mean():.4f}")
        print(f"Router success       = {pivot['router'].mean():.4f}")
        print(f"Oracle best success  = {pivot['oracle_best'].mean():.4f}")
        print(
            f"Router->Oracle gap   = "
            f"{pivot['oracle_best'].mean() - pivot['router'].mean():.4f}"
        )

    # Which fixed strategy actually wins by task_score?
    print("\n" + "=" * 100)
    print("ACTUAL STATIC WINNER BY GROUP")
    print("=" * 100)

    fixed = df[
        df["method"].isin(["always_react", "always_plan"])
    ].pivot_table(
        index=["id", "group"],
        columns="method",
        values="task_score",
        aggfunc="first",
    )

    if {"always_react", "always_plan"}.issubset(set(fixed.columns)):
        def winner(row):
            if row["always_react"] > row["always_plan"]:
                return "REACT"
            if row["always_plan"] > row["always_react"]:
                return "PLAN"
            return "TIE"

        fixed["winner"] = fixed.apply(winner, axis=1)

        outcome = (
            fixed.reset_index()
            .groupby(["group", "winner"])
            .size()
            .rename("n")
            .reset_index()
        )

        outcome["ratio"] = (
            outcome.groupby("group")["n"]
            .transform(lambda s: s / s.sum())
        )

        print(outcome.to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-squad", type=int, default=50)
    parser.add_argument("--n-hotpot", type=int, default=50)
    parser.add_argument("--n-bfcl-single", type=int, default=50)
    parser.add_argument("--n-bfcl-multi", type=int, default=50)

    parser.add_argument("--corpus-size", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3)

    parser.add_argument("--model", type=str, default=MODEL_DEFAULT)

    parser.add_argument(
        "--effort",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
    )

    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)

    parser.add_argument(
        "--output",
        type=str,
        default="unified_dependency_router_results.csv",
    )

    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    use_cache = not args.no_cache

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 100)
    print("CONFIG")
    print("=" * 100)
    print(f"model         : {args.model}")
    print(f"effort        : {args.effort}")
    print(f"SQuAD         : {args.n_squad}")
    print(f"HotpotQA      : {args.n_hotpot}")
    print(f"BFCL single   : {args.n_bfcl_single}")
    print(f"BFCL multi    : {args.n_bfcl_multi}")
    print(f"corpus size   : {args.corpus_size}")
    print(f"top-k         : {args.top_k}")
    print(f"max steps     : {args.max_steps}")
    print(f"cache         : {use_cache}")

    print("\n[LOAD] SQuAD...")
    squad_items, squad_corpus = build_squad(
        args.n_squad,
        args.corpus_size,
        args.seed,
    )

    print("[LOAD] HotpotQA...")
    hotpot_items, hotpot_corpus = build_hotpot(
        args.n_hotpot,
        args.corpus_size,
        args.seed,
    )

    print("[LOAD] BFCL...")
    bfcl_items = build_bfcl(
        args.n_bfcl_single,
        args.n_bfcl_multi,
        args.seed,
    )

    searchers = {
        "squad": BM25Search(squad_corpus),
        "hotpotqa": BM25Search(hotpot_corpus),
    }

    items = squad_items + hotpot_items + bfcl_items

    rng = random.Random(args.seed)
    rng.shuffle(items)

    print("\n" + "=" * 100)
    print("DATASET")
    print("=" * 100)

    counts = pd.Series([x["group"] for x in items]).value_counts()
    print(counts.to_string())

    print("\nDependency labels:")
    print("  SQuAD       -> PLAN (1)")
    print("  HotpotQA    -> REACT (0)")
    print("  BFCL single -> PLAN (1)")
    print("  BFCL multi  -> PLAN (1)")

    methods = [
        "always_react",
        "always_plan",
        "router",
        "static_dependency",
    ]

    rows = []

    for i, item in enumerate(
        tqdm(items, total=len(items), desc="examples"),
        start=1,
    ):
        print("\n" + "=" * 100)
        print(
            f"[EXAMPLE {i}/{len(items)}] "
            f"group={item['group']} family={item['family']} "
            f"gold_route={item['dependency_route']}"
        )
        print("=" * 100)
        print(f"Q: {item['question']}")

        for method in methods:
            print(f"\n  >>> METHOD: {method}")

            try:
                row = evaluate_method(
                    item=item,
                    method=method,
                    retrieval_searchers=searchers,
                    top_k=args.top_k,
                    max_steps=args.max_steps,
                    model=args.model,
                    effort=args.effort,
                    use_cache=use_cache,
                )
                rows.append(row)

            except Exception as e:
                print(
                    f"      [ERROR] id={item['id']} "
                    f"method={method}: {repr(e)}",
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No successful rows.")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print_summary(df)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Saved: {args.output}")
    print(f"Unique cached prompts: {len(LLM_CACHE)}")

    if use_cache:
        print(
            "NOTE: cache ON -> quality comparison is valid, "
            "but Router latency is not standalone production latency. "
            "Use --no-cache for latency/cost comparison."
        )


if __name__ == "__main__":
    main()
