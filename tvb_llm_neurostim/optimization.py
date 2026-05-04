"""LLM-guided optimization loops."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic

from tvb_llm_neurostim.config import ModelConfig, OptimizationConfig, PathsConfig
from tvb_llm_neurostim.json_utils import parse_json_response
from tvb_llm_neurostim.simulation import (
    get_clinical_recommendation,
    get_labels,
    run_robust,
    run_robust_clinical,
)

DEFAULT_MODELS = ModelConfig()
DEFAULT_OPTIMIZATION = OptimizationConfig()
DEFAULT_PATHS = PathsConfig()


def parse_parameter_json(text: str) -> dict[str, Any]:
    parsed = parse_json_response(text, context="optimizer response")
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected optimizer JSON object, got {type(parsed).__name__}")
    return parsed


def propose_intrinsic_parameters(
    client: anthropic.Anthropic,
    history: list[dict[str, Any]],
    *,
    model_config: ModelConfig = DEFAULT_MODELS,
) -> dict[str, float]:
    response = client.messages.create(
        model=model_config.optimizer_model,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": f"""You are optimizing epilepsy intervention parameters for 5 virtual patients.

Goal: maximize WORST-CASE reward across all patients (robust optimization).
Parameters to tune:
- x0: epileptogenicity, range -3.0 to -1.0
- coupling_a: network coupling strength, range 0.005 to 0.030

History:
{json.dumps(history, indent=2)}

Respond ONLY with JSON, no explanation: {{"x0": <float>, "coupling_a": <float>}}""",
            }
        ],
    )
    parsed = parse_parameter_json(response.content[0].text)
    return {"x0": float(parsed["x0"]), "coupling_a": float(parsed["coupling_a"])}


def run_intrinsic_optimization(
    *,
    output_json: Path = DEFAULT_PATHS.results_json,
    api_key: str | None = None,
    optimization_config: OptimizationConfig = DEFAULT_OPTIMIZATION,
) -> list[dict[str, Any]]:
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    history: list[dict[str, Any]] = []
    best_worst = float("-inf")

    for iteration in range(optimization_config.n_iterations):
        if not history:
            params = {"x0": -1.6, "coupling_a": 0.0152}
        else:
            params = propose_intrinsic_parameters(client, history)

        worst, mean, _ = run_robust(params["x0"], coupling_a=params["coupling_a"])
        history.append(
            {
                "iteration": iteration,
                "x0": params["x0"],
                "coupling_a": params["coupling_a"],
                "worst_case_reward": round(worst, 4),
                "mean_reward": round(mean, 4),
            }
        )
        marker = "*** best ***" if worst > best_worst else ""
        best_worst = max(best_worst, worst)
        print(
            f"Iter {iteration}: x0={params['x0']:.3f} coupling={params['coupling_a']:.4f} "
            f"| worst={worst:.4f} | mean={mean:.4f} {marker}"
        )

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"\nBest worst-case reward: {best_worst:.4f}")
    print(f"Saved to {output_json}")
    return history


def propose_clinical_parameters(
    client: anthropic.Anthropic,
    history: list[dict[str, Any]],
    *,
    model_config: ModelConfig = DEFAULT_MODELS,
) -> dict[str, Any]:
    response = client.messages.create(
        model=model_config.optimizer_model,
        max_tokens=model_config.optimization_max_tokens,
        messages=[
            {
                "role": "user",
                "content": f"""You are optimizing brain stimulation parameters for epilepsy treatment across 5 virtual patients.

Goal: maximize WORST-CASE reward (minimax optimization). Higher reward = less seizure activity.

Parameters to optimize:
- iext_boost: stimulation intensity added to Iext (range 0.0 to 4.0, float)
- site_index: brain region to stimulate (integer 0-75)

Key brain regions:
7: rFEF (right frontal eye field)
9: rHC (right hippocampus)
12: rM1 (right motor cortex)
37: rCC (right corpus callosum)
47: lHC (left hippocampus)
50: lM1 (left motor cortex)

History:
{json.dumps(history, indent=2)}

Respond ONLY with JSON: {{"iext_boost": <float>, "site_index": <int>}}""",
            }
        ],
    )
    parsed = parse_parameter_json(response.content[0].text)
    return {"iext_boost": float(parsed["iext_boost"]), "site_index": int(parsed["site_index"])}


def run_clinical_optimization(
    *,
    output_json: Path = DEFAULT_PATHS.clinical_results_json,
    api_key: str | None = None,
    optimization_config: OptimizationConfig = DEFAULT_OPTIMIZATION,
) -> dict[str, Any]:
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    labels = get_labels()
    history: list[dict[str, Any]] = []
    best_worst = float("-inf")
    best_params: dict[str, Any] | None = None

    print("=== Clinical Stimulation Optimization (Minimax) ===\n")
    baseline_worst, baseline_mean, _ = run_robust_clinical(0.0, optimization_config.baseline_site_index)
    print(f"Baseline: worst={baseline_worst:.4f} mean={baseline_mean:.4f}\n")

    for iteration in range(optimization_config.n_iterations):
        if not history:
            params = {
                "iext_boost": optimization_config.initial_iext_boost,
                "site_index": optimization_config.initial_site_index,
            }
        else:
            params = propose_clinical_parameters(client, history)

        site_name = labels[params["site_index"]] if params["site_index"] < len(labels) else "unknown"
        worst, mean, _ = run_robust_clinical(params["iext_boost"], params["site_index"])
        history.append(
            {
                "iteration": iteration,
                "iext_boost": round(params["iext_boost"], 2),
                "site_index": params["site_index"],
                "site_name": site_name,
                "worst_case_reward": round(worst, 4),
                "mean_reward": round(mean, 4),
            }
        )
        marker = "*** best ***" if worst > best_worst else ""
        if worst > best_worst:
            best_worst = worst
            best_params = dict(params)
        print(
            f"Iter {iteration}: boost={params['iext_boost']:.1f} site={params['site_index']}({site_name}) "
            f"| worst={worst:.4f} mean={mean:.4f} {marker}"
        )

    assert best_params is not None
    improvement = (best_worst - baseline_worst) / abs(baseline_worst) * 100
    recommendation = get_clinical_recommendation(
        best_params["iext_boost"],
        best_params["site_index"],
    )
    result = {
        "baseline_worst": round(baseline_worst, 4),
        "best_worst": round(best_worst, 4),
        "improvement_pct": round(improvement, 1),
        "recommendation": recommendation,
        "history": history,
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nClinical recommendation: {recommendation['description']}")
    print(f"Saved to {output_json}")
    return result


def run_bo_comparison(
    *,
    output_json: Path = Path("bo_comparison.json"),
    api_key: str | None = None,
    n_calls: int = 8,
    objective: Callable[[float, int], float] | None = None,
) -> dict[str, Any]:
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real
    except ImportError as exc:
        raise RuntimeError("Install analysis dependencies with `uv sync --extra analysis`.") from exc

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    objective_fn = objective or (lambda boost, site: run_robust_clinical(boost, int(site))[0])
    bo_calls: list[float] = []
    llm_calls: list[float] = []

    def bo_objective(params: list[float]) -> float:
        iext, site = params
        reward = objective_fn(float(iext), int(site))
        bo_calls.append(reward)
        return -reward

    bo_result = gp_minimize(
        bo_objective,
        [Real(0.0, 4.0, name="iext_boost"), Integer(0, 75, name="site_index")],
        n_calls=n_calls,
        n_initial_points=3,
        random_state=42,
        noise=1e-6,
    )

    system_prior = """You are an expert computational neuroscientist.
You have read 136 papers on epilepsy neurostimulation.
Prefer moderate intensities, hippocampal targets for temporal epilepsy, and minimax robustness."""
    history: list[dict[str, Any]] = []
    reasoning_log: list[dict[str, Any]] = []
    for iteration in range(n_calls):
        if not history:
            params = {"iext_boost": 1.5, "site_index": 9}
            reasoning = "Literature prior: rHC at moderate intensity."
        else:
            response = client.messages.create(
                model=DEFAULT_MODELS.optimizer_model,
                max_tokens=512,
                system=system_prior,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"History:\n{json.dumps(history, indent=2)}\n\n"
                            "Return JSON only: "
                            '{"reasoning": "...", "iext_boost": <float>, "site_index": <int>}'
                        ),
                    }
                ],
            )
            parsed = parse_parameter_json(response.content[0].text)
            params = {
                "iext_boost": float(parsed["iext_boost"]),
                "site_index": int(parsed["site_index"]),
            }
            reasoning = re.sub(r"\s+", " ", parsed.get("reasoning", "")).strip()

        reward = objective_fn(params["iext_boost"], params["site_index"])
        llm_calls.append(reward)
        history.append(
            {
                "iteration": iteration,
                "iext_boost": round(params["iext_boost"], 2),
                "site_index": params["site_index"],
                "worst_case_reward": round(reward, 4),
            }
        )
        reasoning_log.append(
            {
                "iteration": iteration,
                "reasoning": reasoning,
                "params": params,
                "reward": round(reward, 4),
            }
        )

    results = {
        "bo_trajectory": [round(value, 4) for value in bo_calls],
        "llm_trajectory": [round(value, 4) for value in llm_calls],
        "bo_best": round(max(bo_calls), 4),
        "llm_best": round(max(llm_calls), 4),
        "iter1_bo": round(bo_calls[0], 4),
        "iter1_llm": round(llm_calls[0], 4),
        "bo_best_params": {"iext_boost": round(float(bo_result.x[0]), 2), "site_index": int(bo_result.x[1])},
        "reasoning_log": reasoning_log,
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved to {output_json}")
    return results
