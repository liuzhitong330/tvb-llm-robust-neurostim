import json
import numpy as np
from scipy import stats
from cohort import PatientCohort
from engine import BatchSimulator

def main():
    cohort_gen = PatientCohort(connectivity_variance=0.1, seed=42)
    cohort = cohort_gen.generate(n_patients=5)

    BEST_IEXT = 0.6
    BEST_SITE = 9  # rHC

    print("\n--- Baseline (no stimulation) ---")
    engine_base = BatchSimulator(checkpoint_file="checkpoint_baseline.json")
    baseline_results = engine_base.run_cohort_study(
        cohort, iext_boost=0.0, site_index=9, max_workers=2)

    print("\n--- Optimized (rHC, boost=0.6) ---")
    engine_opt = BatchSimulator(checkpoint_file="checkpoint_optimized.json")
    opt_results = engine_opt.run_cohort_study(
        cohort, iext_boost=BEST_IEXT, site_index=BEST_SITE, max_workers=2)

    base_rewards = [r["reward"] for r in baseline_results if r["status"] == "success"]
    opt_rewards = [r["reward"] for r in opt_results if r["status"] == "success"]

    print(f"\n=== Results (n={len(base_rewards)}) ===")
    print(f"Baseline:  mean={np.mean(base_rewards):.4f} std={np.std(base_rewards):.4f}")
    print(f"Optimized: mean={np.mean(opt_rewards):.4f} std={np.std(opt_rewards):.4f}")

    t, p = stats.ttest_rel(opt_rewards, base_rewards)
    d = (np.mean(opt_rewards) - np.mean(base_rewards)) / np.std(base_rewards)
    print(f"Paired t-test: t={t:.3f}, p={p:.4f}")
    print(f"Cohen's d: {d:.3f}")

    improvements = [o - b for o, b in zip(opt_rewards, base_rewards)]
    responders = sum(1 for i in improvements if i > 0)
    print(f"Responders: {responders}/{len(improvements)} ({100*responders/len(improvements):.0f}%)")

    with open("cohort_results_test.json", "w") as f:
        json.dump({
            "baseline": baseline_results,
            "optimized": opt_results,
            "stats": {"t": round(t,3), "p": round(p,4), 
                      "cohens_d": round(d,3),
                      "responder_rate": round(responders/len(improvements),2)}
        }, f, indent=2)
    print("\nSaved to cohort_results_test.json")

if __name__ == "__main__":
    main()
