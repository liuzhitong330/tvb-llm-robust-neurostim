"""External-stimulation landscape experiments.

This module runs the bounded all-region stimulation sweep used by the paper to
calibrate the LLM, Bayesian-optimization, random-search, and heuristic results
against a common evaluated landscape.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tvb_llm_neurostim.config import PathsConfig
from tvb_llm_neurostim.simulation import get_labels, run_robust_clinical

DEFAULT_BOOSTS = (0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0)
DEFAULT_RANDOM_SEEDS = 50
DEFAULT_RANDOM_BUDGET = 8
DEFAULT_PATHS = PathsConfig()


@dataclass(frozen=True)
class Candidate:
    """One external-stimulation candidate."""

    iext_boost: float
    site_index: int


def _evaluate_candidate(candidate: Candidate) -> dict[str, Any]:
    warnings.filterwarnings("ignore", message="Geodesic distance module is unavailable.*")
    logging.getLogger("tvb").setLevel(logging.ERROR)
    logging.getLogger("tvb.basic.readers").setLevel(logging.ERROR)
    logging.getLogger("tvb.simulator.integrators").setLevel(logging.ERROR)
    worst, mean, rewards = run_robust_clinical(candidate.iext_boost, candidate.site_index)
    labels = get_labels()
    return {
        "iext_boost": candidate.iext_boost,
        "site_index": candidate.site_index,
        "site_name": labels[candidate.site_index],
        "worst_case_reward": worst,
        "mean_reward": mean,
        "rewards": rewards,
    }


def _round_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "iext_boost": round(float(row["iext_boost"]), 2),
        "site_index": int(row["site_index"]),
        "site_name": row["site_name"],
        "worst_case_reward": round(float(row["worst_case_reward"]), 4),
        "mean_reward": round(float(row["mean_reward"]), 4),
        "rewards": [round(float(value), 4) for value in row["rewards"]],
    }


def run_stimulation_landscape(
    *,
    output_json: Path = DEFAULT_PATHS.clinical_landscape_json,
    boosts: tuple[float, ...] = DEFAULT_BOOSTS,
    max_workers: int = 4,
    random_seeds: int = DEFAULT_RANDOM_SEEDS,
    random_budget: int = DEFAULT_RANDOM_BUDGET,
) -> dict[str, Any]:
    """Evaluate all 76 regions over the selected boost grid and summarize baselines."""

    labels = get_labels()
    candidates = [
        Candidate(iext_boost=boost, site_index=site_index)
        for site_index in range(len(labels))
        for boost in boosts
    ]
    print(
        f"Evaluating {len(candidates)} candidates "
        f"({len(labels)} sites x {len(boosts)} boosts) with {max_workers} workers."
    )

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_evaluate_candidate, candidate): candidate for candidate in candidates}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if index % 25 == 0 or index == len(candidates):
                best_so_far = max(rows, key=lambda item: item["worst_case_reward"])
                print(
                    f"{index:>4}/{len(candidates)} complete | "
                    f"best={best_so_far['site_name']} b={best_so_far['iext_boost']:.2f} "
                    f"worst={best_so_far['worst_case_reward']:.4f}"
                )

    rounded_rows = [_round_candidate(row) for row in rows]
    rounded_rows.sort(key=lambda row: (row["site_index"], row["iext_boost"]))
    best = max(rounded_rows, key=lambda row: row["worst_case_reward"])
    baseline = next(
        row
        for row in rounded_rows
        if row["site_index"] == 9 and row["iext_boost"] == 0.0
    )

    rng = np.random.default_rng(42)
    random_best: list[float] = []
    random_best_rows: list[dict[str, Any]] = []
    candidate_indices = np.arange(len(rounded_rows))
    for _seed_index in range(random_seeds):
        sampled = rng.choice(candidate_indices, size=random_budget, replace=False)
        sampled_rows = [rounded_rows[int(index)] for index in sampled]
        best_sample = max(sampled_rows, key=lambda row: row["worst_case_reward"])
        random_best.append(float(best_sample["worst_case_reward"]))
        random_best_rows.append(best_sample)

    heuristic_sites = {
        "right_hippocampus": 9,
        "left_hippocampus": 47,
        "right_fef_hub": 7,
        "right_pfc_orb_top_hub": 21,
    }
    heuristic_summary = {}
    for name, site_index in heuristic_sites.items():
        site_rows = [row for row in rounded_rows if row["site_index"] == site_index]
        heuristic_summary[name] = max(site_rows, key=lambda row: row["worst_case_reward"])

    summary = {
        "boosts": list(boosts),
        "n_sites": len(labels),
        "n_candidates": len(rounded_rows),
        "n_patients_per_candidate": 5,
        "baseline": baseline,
        "grid_best": best,
        "top10": sorted(
            rounded_rows,
            key=lambda row: row["worst_case_reward"],
            reverse=True,
        )[:10],
        "random_search": {
            "n_seeds": random_seeds,
            "budget": random_budget,
            "median_best_reward": round(float(np.median(random_best)), 4),
            "mean_best_reward": round(float(np.mean(random_best)), 4),
            "p90_best_reward": round(float(np.quantile(random_best, 0.9)), 4),
            "max_best_reward": round(float(np.max(random_best)), 4),
            "best_rewards": [round(value, 4) for value in random_best],
            "best_candidates": random_best_rows,
        },
        "heuristics": heuristic_summary,
        "grid": rounded_rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved {output_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATHS.clinical_landscape_json)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--random-seeds", type=int, default=DEFAULT_RANDOM_SEEDS)
    parser.add_argument(
        "--boosts",
        type=float,
        nargs="*",
        default=list(DEFAULT_BOOSTS),
        help="Boost values to evaluate. Defaults to the paper grid.",
    )
    args = parser.parse_args()
    run_stimulation_landscape(
        output_json=args.output,
        boosts=tuple(args.boosts),
        max_workers=args.workers,
        random_seeds=args.random_seeds,
    )


if __name__ == "__main__":
    main()
