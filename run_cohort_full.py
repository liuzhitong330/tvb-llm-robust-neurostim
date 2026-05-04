#!/usr/bin/env python3
"""Run the 20-patient virtual cohort study."""

from __future__ import annotations

import argparse
from pathlib import Path

from tvb_llm_neurostim.cohort_analysis import run_cohort_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Delete checkpoints before rerun.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    checkpoints = [Path("checkpoint_baseline_20.json"), Path("checkpoint_optimized_20.json")]
    if args.force:
        for checkpoint in checkpoints:
            checkpoint.unlink(missing_ok=True)

    summary = run_cohort_study(
        n_patients=20,
        output_json=Path("cohort_results_20.json"),
        baseline_checkpoint=checkpoints[0],
        optimized_checkpoint=checkpoints[1],
        max_workers=args.workers,
    )
    print(f"\n=== Results (n={summary['n_patients']}) ===")
    print(f"Baseline:  mean={summary['baseline']['mean']:.4f} std={summary['baseline']['std']:.4f}")
    print(f"Optimized: mean={summary['optimized']['mean']:.4f} std={summary['optimized']['std']:.4f}")
    print(
        f"Paired t-test: t={summary['stats']['t']:.3f}, "
        f"p={summary['stats']['p']:.4f}"
    )
    print(f"Cohen's d: {summary['stats']['cohens_d']:.3f}")
    print(f"Responder rate: {summary['stats']['responder_rate']:.0%}")
    print(f"Paradoxical rate: {summary['stats']['paradoxical_rate']:.0%}")


if __name__ == "__main__":
    main()
