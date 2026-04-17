import anthropic
import json
import re
import sys
sys.path.insert(0, '/Users/apple/Desktop/VBT/rl_project')
from simulate import run_robust

client = anthropic.Anthropic()

def llm_propose(history):
    history_str = json.dumps(history, indent=2)
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""You are optimizing epilepsy intervention parameters for 5 virtual patients.

Goal: maximize WORST-CASE reward across all patients (robust optimization).
Parameters to tune:
- x0: epileptogenicity, range -3.0 to -1.0
- coupling_a: network coupling strength, range 0.005 to 0.030

History:
{history_str}

Respond ONLY with JSON, no explanation: {{"x0": <float>, "coupling_a": <float>}}"""
        }]
    )
    text = response.content[0].text.strip()
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Cannot parse: {text}")

history = []
best_worst = float('-inf')

for i in range(8):
    if not history:
        params = {"x0": -1.6, "coupling_a": 0.0152}  # baseline
    else:
        params = llm_propose(history)

    worst, mean, all_rewards = run_robust(params["x0"], n_patients=5)
    history.append({
        "iteration": i,
        "x0": params["x0"],
        "coupling_a": params.get("coupling_a", 0.0152),
        "worst_case_reward": round(worst, 4),
        "mean_reward": round(mean, 4)
    })

    marker = "*** best ***" if worst > best_worst else ""
    if worst > best_worst:
        best_worst = worst
    print(f"Iter {i}: x0={params['x0']:.3f} coupling={params.get('coupling_a', 0.0152):.4f} | worst={worst:.4f} | mean={mean:.4f} {marker}")

print(f"\nBest worst-case reward: {best_worst:.4f}")

import json
with open("results.json", "w") as f:
    json.dump(history, f, indent=2)
print("Saved to results.json")
