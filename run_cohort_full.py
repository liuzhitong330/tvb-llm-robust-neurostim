import json
import numpy as np
from scipy import stats
from cohort import PatientCohort
from engine import BatchSimulator

def main():
    # 删掉旧checkpoint，重新跑20个病人
    import os
    for f in ["checkpoint_baseline_20.json", "checkpoint_optimized_20.json"]:
        if os.path.exists(f): os.remove(f)

    cohort_gen = PatientCohort(connectivity_variance=0.1, seed=42)
    cohort = cohort_gen.generate(n_patients=20)

    print("\n--- Baseline ---")
    engine_base = BatchSimulator(checkpoint_file="checkpoint_baseline_20.json")
    baseline_results = engine_base.run_cohort_study(
        cohort, iext_boost=0.0, site_index=9, max_workers=4)

    print("\n--- Optimized (rHC, boost=0.6) ---")
    engine_opt = BatchSimulator(checkpoint_file="checkpoint_optimized_20.json")
    opt_results = engine_opt.run_cohort_study(
        cohort, iext_boost=0.6, site_index=9, max_workers=4)

    base_rewards = [r["reward"] for r in baseline_results if r["status"] == "success"]
    opt_rewards = [r["reward"] for r in opt_results if r["status"] == "success"]
    base_soz = [r["soz_type"] for r in baseline_results if r["status"] == "success"]

    print(f"\n=== Results (n={len(base_rewards)}) ===")
    print(f"Baseline:  mean={np.mean(base_rewards):.4f} std={np.std(base_rewards):.4f}")
    print(f"Optimized: mean={np.mean(opt_rewards):.4f} std={np.std(opt_rewards):.4f}")

    t, p = stats.ttest_rel(opt_rewards, base_rewards)
    d = (np.mean(opt_rewards) - np.mean(base_rewards)) / np.std(base_rewards)
    print(f"Paired t-test: t={t:.3f}, p={p:.4f}")
    print(f"Cohen's d: {d:.3f}")

    improvements = [o - b for o, b in zip(opt_rewards, base_rewards)]
    responders = sum(1 for i in improvements if i > 0)
    paradoxical = sum(1 for i in improvements if i < -0.01)
    print(f"Responders: {responders}/{len(improvements)} ({100*responders/len(improvements):.0f}%)")
    print(f"Paradoxical responders: {paradoxical}/{len(improvements)} ({100*paradoxical/len(improvements):.0f}%)")

    # SOZ subgroup analysis
    print("\n--- SOZ Subgroup Analysis ---")
    soz_types = set(base_soz)
    for soz in soz_types:
        idx = [i for i, s in enumerate(base_soz) if s == soz]
        b = [base_rewards[i] for i in idx]
        o = [opt_rewards[i] for i in idx]
        imp = np.mean([o[j]-b[j] for j in range(len(b))])
        print(f"  {soz}: n={len(b)}, avg improvement={imp:.4f}")

    final = {
        "n_patients": len(base_rewards),
        "baseline": {"mean": round(np.mean(base_rewards),4), "std": round(np.std(base_rewards),4),
                     "rewards": base_rewards},
        "optimized": {"mean": round(np.mean(opt_rewards),4), "std": round(np.std(opt_rewards),4),
                      "rewards": opt_rewards},
        "stats": {"t": round(t,3), "p": round(p,4), "cohens_d": round(d,3),
                  "responder_rate": round(responders/len(improvements),2),
                  "paradoxical_rate": round(paradoxical/len(improvements),2)},
        "soz_types": base_soz
    }
    with open("cohort_results_20.json", "w") as f:
        json.dump(final, f, indent=2)
    print("\nSaved to cohort_results_20.json")

if __name__ == "__main__":
    main()
