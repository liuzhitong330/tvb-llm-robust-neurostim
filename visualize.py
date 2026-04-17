import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

with open("results.json") as f:
    history = json.load(f)

iters = [h["iteration"] for h in history]
worst = [h["worst_case_reward"] for h in history]
mean = [h["mean_reward"] for h in history]
x0s = [h["x0"] for h in history]
couplings = [h["coupling_a"] for h in history]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 图1：reward随迭代变化
axes[0].plot(iters, worst, 'o-', color='crimson', label='Worst-case reward', linewidth=2)
axes[0].plot(iters, mean, 's--', color='steelblue', label='Mean reward', linewidth=2)
axes[0].axhline(y=worst[0], color='gray', linestyle=':', label='Baseline worst-case')
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Reward")
axes[0].set_title("Robust Optimization: Reward over Iterations")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2：参数空间探索
scatter = axes[1].scatter(x0s, couplings, c=worst, cmap='RdYlGn', 
                           s=150, zorder=5, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, ax=axes[1], label='Worst-case reward')
axes[1].set_xlabel("x0 (epileptogenicity)")
axes[1].set_ylabel("coupling_a")
axes[1].set_title("Parameter Space Exploration")
for i, h in enumerate(history):
    axes[1].annotate(str(i), (h["x0"], h["coupling_a"]), 
                     textcoords="offset points", xytext=(5,5), fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches='tight')
print("Saved results.png")
plt.show()
