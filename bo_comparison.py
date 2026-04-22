"""
LLM Optimizer vs Bayesian Optimization: Convergence Comparison
Both optimizers search the same (iext_boost, site_index) space.
"""
import json
import numpy as np
import anthropic
import re
from skopt import gp_minimize
from skopt.space import Real, Integer
from simulate_v2 import run_robust_clinical

client = anthropic.Anthropic()

# ── shared objective ──────────────────────────────────────────────────────────
def objective(iext_boost, site_index):
    worst, mean, _ = run_robust_clinical(iext_boost, int(site_index))
    return worst          # maximise → we negate inside BO

bo_calls = []
llm_calls = []

# ── Bayesian Optimisation (blind, no domain knowledge) ────────────────────────
print("=" * 60)
print("BAYESIAN OPTIMISATION  (GP, random init, no prior knowledge)")
print("=" * 60)

def bo_objective(params):
    iext, site = params
    reward = objective(iext, int(site))
    bo_calls.append(reward)
    best_so_far = max(bo_calls)
    print(f"  BO iter {len(bo_calls):2d}: boost={iext:.2f} site={int(site):2d} "
          f"reward={reward:.4f}  best={best_so_far:.4f}")
    return -reward        # skopt minimises

space = [Real(0.0, 4.0, name="iext_boost"),
         Integer(0, 75,  name="site_index")]

bo_result = gp_minimize(
    bo_objective,
    space,
    n_calls=8,
    n_initial_points=3,   # 3 random, then GP-guided
    random_state=42,
    noise=1e-6,
)
bo_best = max(bo_calls)
print(f"\nBO best reward: {bo_best:.4f}")
print(f"BO best params: boost={bo_result.x[0]:.2f}, site={bo_result.x[1]}")

# ── LLM Optimiser (informed by Stage 1 literature mining) ────────────────────
print("\n" + "=" * 60)
print("LLM OPTIMISER  (Claude Opus, Stage-1 literature prior)")
print("=" * 60)

SYSTEM_PRIOR = """You are an expert computational neuroscientist.
You have read 136 papers on epilepsy neurostimulation (Stage 1 literature mining).
Key findings from the literature:
- Temporal lobe epilepsy often responds to hippocampal stimulation (rHC = index 9, lHC = index 47)
- Frontal lobe targets (rFEF = index 7) show moderate efficacy
- High stimulation intensities (>3.0) carry risk of paradoxical seizure induction
- Moderate intensities (0.5–2.0) are typically safer and more robust across patients
- The goal is minimax optimisation: maximise the WORST-CASE reward across 5 virtual patients
"""

history = []
reasoning_log = []

for i in range(8):
    if not history:
        # Iteration 0: literature-informed first guess
        params = {"iext_boost": 1.5, "site_index": 9}
        reasoning = "Literature prior: rHC (index 9) is the most evidence-supported target for temporal lobe epilepsy. Starting with moderate intensity 1.5 to balance efficacy and safety."
    else:
        history_str = json.dumps(history, indent=2)
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=SYSTEM_PRIOR,
            messages=[{"role": "user", "content": f"""
History of evaluated parameters:
{history_str}

Based on neuroscience literature and observed results, propose the next 
(iext_boost, site_index) pair to maximise worst-case reward across patients.

Constraints:
- iext_boost: 0.0 to 4.0
- site_index: 0 to 75 (integer)
- Avoid intensity > 3.0 (safety risk)

Reason step by step, then respond with JSON:
{{"reasoning": "<your clinical reasoning>", "iext_boost": <float>, "site_index": <int>}}
"""}]
        )
        text = response.content[0].text.strip()
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        parsed = json.loads(match.group())
        params = {"iext_boost": float(parsed["iext_boost"]),
                  "site_index": int(parsed["site_index"])}
        reasoning = parsed.get("reasoning", "")

    reward = objective(params["iext_boost"], params["site_index"])
    llm_calls.append(reward)
    best_so_far = max(llm_calls)

    history.append({
        "iteration": i,
        "iext_boost": round(params["iext_boost"], 2),
        "site_index": params["site_index"],
        "worst_case_reward": round(reward, 4),
    })
    reasoning_log.append({
        "iteration": i,
        "reasoning": reasoning,
        "params": params,
        "reward": round(reward, 4),
    })

    print(f"  LLM iter {i:2d}: boost={params['iext_boost']:.2f} "
          f"site={params['site_index']:2d} reward={reward:.4f}  best={best_so_far:.4f}")

llm_best = max(llm_calls)
print(f"\nLLM best reward: {llm_best:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Metric':<35} {'BO':>10} {'LLM':>10}")
print("-" * 55)
print(f"{'Best reward (8 iterations)':<35} {bo_best:>10.4f} {llm_best:>10.4f}")

# First time exceeding -0.50 (proxy for "good" solution)
threshold = -0.50
bo_first = next((i+1 for i, r in enumerate(bo_calls) if r > threshold), None)
llm_first = next((i+1 for i, r in enumerate(llm_calls) if r > threshold), None)
print(f"{'Iters to exceed −0.50':<35} {str(bo_first or 'never'):>10} {str(llm_first or 'never'):>10}")
print(f"{'Iter-1 reward (cold start)':<35} {bo_calls[0]:>10.4f} {llm_calls[0]:>10.4f}")

results = {
    "bo_trajectory": [round(r, 4) for r in bo_calls],
    "llm_trajectory": [round(r, 4) for r in llm_calls],
    "bo_best": round(bo_best, 4),
    "llm_best": round(llm_best, 4),
    "iter1_bo": round(bo_calls[0], 4),
    "iter1_llm": round(llm_calls[0], 4),
    "reasoning_log": reasoning_log,
}
with open("bo_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to bo_comparison.json")
