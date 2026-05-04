#!/usr/bin/env python3
"""Run the small cohort smoke study."""

from pathlib import Path

from tvb_llm_neurostim.cohort_analysis import run_cohort_study


def main() -> None:
    summary = run_cohort_study(
        n_patients=5,
        output_json=Path("cohort_results_test.json"),
        baseline_checkpoint=Path("checkpoint_baseline.json"),
        optimized_checkpoint=Path("checkpoint_optimized.json"),
        max_workers=2,
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


if __name__ == "__main__":
    main()
