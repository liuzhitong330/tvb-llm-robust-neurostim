"""Cohort study orchestration and statistics."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from tvb_llm_neurostim.batch import BatchSimulator
from tvb_llm_neurostim.cohort import PatientCohort
from tvb_llm_neurostim.config import CohortConfig, OptimizationConfig

DEFAULT_COHORT = CohortConfig()
DEFAULT_OPTIMIZATION = OptimizationConfig()


def paired_ttest(opt_rewards: list[float], base_rewards: list[float]) -> tuple[float, float]:
    """Return paired t and two-sided p value, using SciPy when available."""

    try:
        from scipy import stats

        t_value, p_value = stats.ttest_rel(opt_rewards, base_rewards)
        return float(t_value), float(p_value)
    except ImportError:
        diffs = np.array(opt_rewards) - np.array(base_rewards)
        if len(diffs) < 2 or float(np.std(diffs, ddof=1)) == 0.0:
            return 0.0, 1.0
        t_value = float(np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs))))
        # Normal approximation fallback keeps the analysis script runnable without SciPy.
        p_value = 2.0 * (1.0 - NormalDist().cdf(abs(t_value)))
        return t_value, p_value


def summarize_rewards(
    baseline_results: list[dict[str, Any]],
    optimized_results: list[dict[str, Any]],
) -> dict[str, Any]:
    base_rewards = [row["reward"] for row in baseline_results if row["status"] == "success"]
    opt_rewards = [row["reward"] for row in optimized_results if row["status"] == "success"]
    if len(base_rewards) != len(opt_rewards):
        raise ValueError("Baseline and optimized result counts differ.")
    if not base_rewards:
        raise ValueError("No successful cohort simulations to summarize.")

    improvements = [
        optimized - baseline
        for optimized, baseline in zip(opt_rewards, base_rewards, strict=True)
    ]
    t_value, p_value = paired_ttest(opt_rewards, base_rewards)
    base_std = float(np.std(base_rewards))
    cohens_d = (
        0.0
        if base_std == 0.0
        else (float(np.mean(opt_rewards)) - float(np.mean(base_rewards))) / base_std
    )
    responders = sum(improvement > 0 for improvement in improvements)
    paradoxical = sum(improvement < -0.01 for improvement in improvements)
    return {
        "n_patients": len(base_rewards),
        "baseline": {
            "mean": round(float(np.mean(base_rewards)), 4),
            "std": round(base_std, 4),
            "rewards": base_rewards,
        },
        "optimized": {
            "mean": round(float(np.mean(opt_rewards)), 4),
            "std": round(float(np.std(opt_rewards)), 4),
            "rewards": opt_rewards,
        },
        "stats": {
            "t": round(t_value, 3),
            "p": round(p_value, 4),
            "cohens_d": round(cohens_d, 3),
            "responder_rate": round(responders / len(improvements), 2),
            "paradoxical_rate": round(paradoxical / len(improvements), 2),
        },
        "soz_types": [row["soz_type"] for row in baseline_results if row["status"] == "success"],
    }


def run_cohort_study(
    *,
    n_patients: int,
    output_json: Path,
    baseline_checkpoint: Path,
    optimized_checkpoint: Path,
    max_workers: int,
    cohort_config: CohortConfig = DEFAULT_COHORT,
    optimization_config: OptimizationConfig = DEFAULT_OPTIMIZATION,
) -> dict[str, Any]:
    cohort = PatientCohort(cohort_config).generate(n_patients=n_patients)

    print("\n--- Baseline ---")
    baseline_results = BatchSimulator(baseline_checkpoint).run_cohort_study(
        cohort,
        iext_boost=0.0,
        site_index=optimization_config.baseline_site_index,
        max_workers=max_workers,
    )

    print("\n--- Optimized (rHC, boost=0.6) ---")
    optimized_results = BatchSimulator(optimized_checkpoint).run_cohort_study(
        cohort,
        iext_boost=optimization_config.best_clinical_iext_boost,
        site_index=optimization_config.best_clinical_site_index,
        max_workers=max_workers,
    )

    summary = summarize_rewards(baseline_results, optimized_results)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved to {output_json}")
    return summary
