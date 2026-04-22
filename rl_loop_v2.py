import anthropic
import json
import re
import sys
sys.path.insert(0, '/Users/apple/Desktop/neurostim_project')
from simulate_v2 import run_robust_clinical, get_clinical_recommendation, get_labels

client = anthropic.Anthropic()
labels = get_labels()

def llm_propose(history):
    history_str = json.dumps(history, indent=2)
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""You are optimizing brain stimulation parameters for epilepsy treatment across 5 virtual patients.

Goal: maximize WORST-CASE reward (minimax optimization). Higher reward = less seizure activity.

Parameters to optimize:
- iext_boost: stimulation intensity added to Iext (range 0.0 to 4.0, float)
- site_index: brain region to stimulate (integer 0-75)

Key brain regions (index: name):
7: rFEF (right frontal eye field)
9: rHC (right hippocampus)
12: rM1 (right motor cortex)
37: rCC (right corpus callosum)
47: lHC (left hippocampus)
50: lM1 (left motor cortex)

History of attempts:
{history_str}

Based on the history, propose the next parameter set to try.
Consider: which sites improved reward? Does higher intensity always help?
Respond ONLY with JSON: {{"iext_boost": <float>, "site_index": <int>}}"""
        }]
    )
    text = response.content[0].text.strip()
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Cannot parse: {text}")

history = []
best_worst = float('-inf')
best_params = None

print("=== Clinical Stimulation Optimization (Minimax) ===\n")
print(f"Baseline (no stim): running...")

# baseline
r0, m0, _ = run_robust_clinical(0.0, 9)
print(f"Baseline: worst={r0:.4f} mean={m0:.4f}\n")

for i in range(8):
    if not history:
        params = {"iext_boost": 2.0, "site_index": 7}
    else:
        params = llm_propose(history)

    iext = float(params["iext_boost"])
    site = int(params["site_index"])
    site_name = labels[site] if site < len(labels) else f"Region {site}"

    worst, mean, _ = run_robust_clinical(iext, site)
    
    history.append({
        "iteration": i,
        "iext_boost": round(iext, 2),
        "site_index": site,
        "site_name": site_name,
        "worst_case_reward": round(worst, 4),
        "mean_reward": round(mean, 4)
    })

    marker = "*** best ***" if worst > best_worst else ""
    if worst > best_worst:
        best_worst = worst
        best_params = params.copy()
    
    print(f"Iter {i}: boost={iext:.1f} site={site}({site_name}) | worst={worst:.4f} mean={mean:.4f} {marker}")

print(f"\n=== RESULT ===")
print(f"Best worst-case reward: {best_worst:.4f}")
print(f"Improvement over baseline: {((best_worst - r0) / abs(r0) * 100):.1f}%")

rec = get_clinical_recommendation(best_params["iext_boost"], best_params["site_index"])
print(f"\nClinical recommendation: {rec['description']}")
print(f"Site: {rec['site']}")

with open("results_v2.json", "w") as f:
    json.dump({"baseline_worst": round(r0, 4), "best_worst": best_worst, 
               "improvement_pct": round((best_worst - r0)/abs(r0)*100, 1),
               "recommendation": rec, "history": history}, f, indent=2)
print("\nSaved to results_v2.json")
