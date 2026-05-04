"""RAG-augmented optimizer using literature gaps as retrieval context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic

from tvb_llm_neurostim.config import ModelConfig, PathsConfig
from tvb_llm_neurostim.json_utils import parse_json_response
from tvb_llm_neurostim.simulation import get_labels, run_robust_clinical

DEFAULT_MODELS = ModelConfig()
DEFAULT_PATHS = PathsConfig()


SITE_KEYWORDS = {
    "rHC": ["hippocampus", "hippocampal", "temporal lobe", "limbic"],
    "lHC": ["hippocampus", "hippocampal", "temporal lobe", "limbic"],
    "rFEF": ["frontal", "prefrontal", "motor cortex"],
    "rTCS": ["temporal", "cortex", "neocortical"],
}


def load_knowledge_base(gaps_json: Path = DEFAULT_PATHS.gaps_json) -> list[dict[str, str]]:
    with gaps_json.open(encoding="utf-8") as handle:
        papers = json.load(handle)
    knowledge_base: list[dict[str, str]] = []
    for paper in papers:
        for question in paper.get("gaps", {}).get("open_questions", []):
            knowledge_base.append(
                {
                    "pmid": str(paper["pmid"]),
                    "title": paper["title"],
                    "question": question,
                }
            )
    print(f"Knowledge base: {len(knowledge_base)} entries from {len(papers)} papers")
    return knowledge_base


def retrieve_knowledge(
    knowledge_base: list[dict[str, str]],
    site_name: str,
    iext_boost: float,
    last_reward_trend: str,
    *,
    k: int = 3,
) -> list[dict[str, str]]:
    keywords = list(SITE_KEYWORDS.get(site_name, ["stimulation", "epilepsy"]))
    if last_reward_trend == "worsening":
        keywords.extend(["paradoxical", "worsening", "adverse", "contraindication"])
    elif last_reward_trend == "plateau":
        keywords.extend(["frequency", "amplitude", "optimization", "parameter"])
    else:
        keywords.extend(["suppression", "inhibition", "efficacy"])

    if iext_boost > 2.5:
        keywords.extend(["high frequency", "depolarization block", "safety"])
    elif iext_boost < 0.5:
        keywords.extend(["low intensity", "subthreshold", "minimal"])

    scored = []
    for entry in knowledge_base:
        text = f"{entry['title']} {entry['question']}".lower()
        score = sum(keyword.lower() in text for keyword in keywords)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored[:k]]


def infer_reward_trend(history: list[dict[str, Any]]) -> str:
    recent = [row["reward"] for row in history[-3:]]
    if len(recent) < 2:
        return "improving"
    if recent[-1] > recent[-2]:
        return "improving"
    if recent[-1] < recent[-2] - 0.002:
        return "worsening"
    return "plateau"


def rag_optimize(
    *,
    gaps_json: Path = DEFAULT_PATHS.gaps_json,
    output_json: Path = Path("rag_results.json"),
    api_key: str | None = None,
    n_iterations: int = 8,
) -> dict[str, Any]:
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    labels = get_labels()
    knowledge_base = load_knowledge_base(gaps_json)
    history: list[dict[str, Any]] = []
    rag_log: list[dict[str, Any]] = []
    best_reward = float("-inf")
    best_params: dict[str, Any] | None = None

    print("=== RAG-Augmented LLM Optimizer ===\n")
    baseline, _, _ = run_robust_clinical(0.0, 9)
    print(f"Baseline: {baseline:.4f}\n")

    for iteration in range(n_iterations):
        if iteration == 0:
            params = {"iext_boost": 1.5, "site_index": 9}
        else:
            trend = infer_reward_trend(history)
            site_name = labels[history[-1]["site_index"]]
            retrieved = retrieve_knowledge(
                knowledge_base,
                site_name,
                history[-1]["iext_boost"],
                trend,
            )
            rag_context = "\n".join(
                f"[PMID {row['pmid']}] {row['title'][:60]}...\n  Gap: {row['question'][:120]}"
                for row in retrieved
            )
            history_context = json.dumps(
                [
                    {
                        "iter": row["iteration"],
                        "boost": row["iext_boost"],
                        "site": row["site_name"],
                        "reward": row["reward"],
                    }
                    for row in history
                ],
                indent=2,
            )
            response = client.messages.create(
                model=DEFAULT_MODELS.optimizer_model,
                max_tokens=600,
                messages=[
                    {
                        "role": "user",
                        "content": f"""You are optimizing brain stimulation parameters for epilepsy treatment.

[QUANTITATIVE FEEDBACK - TVB Simulation Results]
Baseline reward: {baseline:.4f}
Optimization history:
{history_context}
Current trend: {trend}

[QUALITATIVE FEEDBACK - Retrieved Medical Knowledge]
{rag_context}

[NETWORK TOPOLOGY CONTEXT]
rHC (index 9): degree=1, strength=2.0 -- isolated node
rFEF (index 7): degree=26, strength=45.0 -- network hub

Respond with JSON only:
{{"reasoning": "<cite specific literature insight>", "iext_boost": <float>, "site_index": <int>}}""",
                    }
                ],
            )
            parsed = parse_json_response(response.content[0].text, context="rag optimizer")
            params = {
                "iext_boost": float(parsed["iext_boost"]),
                "site_index": int(parsed["site_index"]),
            }
            rag_log.append(
                {
                    "iteration": iteration,
                    "trend": trend,
                    "retrieved_papers": [
                        {
                            "pmid": row["pmid"],
                            "title": row["title"][:80],
                            "gap": row["question"][:120],
                        }
                        for row in retrieved
                    ],
                    "reasoning": parsed.get("reasoning", ""),
                    "params": params,
                }
            )

        reward, mean, _ = run_robust_clinical(params["iext_boost"], params["site_index"])
        del mean
        if reward > best_reward:
            best_reward = reward
            best_params = dict(params)
        site_name = labels[params["site_index"]]
        history.append(
            {
                "iteration": iteration,
                "iext_boost": round(params["iext_boost"], 2),
                "site_index": params["site_index"],
                "site_name": site_name,
                "reward": round(reward, 4),
            }
        )
        print(
            f"Iter {iteration}: boost={params['iext_boost']:.2f} site={params['site_index']}({site_name}) "
            f"| reward={reward:.4f} best={best_reward:.4f}"
        )

    assert best_params is not None
    improvement = (best_reward - baseline) / abs(baseline) * 100
    result = {
        "baseline": round(baseline, 4),
        "best_reward": round(best_reward, 4),
        "improvement_pct": round(improvement, 1),
        "best_params": best_params,
        "history": history,
        "rag_log": rag_log,
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved to {output_json}")
    return result
