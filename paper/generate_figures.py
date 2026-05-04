#!/usr/bin/env python3
"""Generate publication figures from checked-in result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)


COLORS = {
    "blue": "#2563eb",
    "teal": "#0d9488",
    "green": "#16a34a",
    "orange": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "gray": "#64748b",
    "dark": "#0f172a",
}


def load_json(name: str):
    with (ROOT / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "savefig.dpi": 300,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def method_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.03, 0.23, 0.25, 0.5, "Literature mining", "136 PubMed papers\n1,080 extracted ideas\nranked research gaps"),
        (0.375, 0.23, 0.25, 0.5, "Virtual cohort", "TVB Epileptor model\n76-region connectome\nsampled patient variability"),
        (0.72, 0.23, 0.25, 0.5, "Minimax search", "LLM proposes parameters\nTVB returns rewards\nmaximize worst case"),
    ]
    for x, y, w, h, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.2,
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
        )
        ax.add_patch(patch)
        ax.text(x + 0.025, y + h - 0.11, title, weight="bold", color=COLORS["dark"], fontsize=10)
        ax.text(x + 0.025, y + h - 0.22, body, color="#475569", va="top", linespacing=1.45)

    for start, end in [((0.29, 0.48), (0.365, 0.48)), ((0.635, 0.48), (0.71, 0.48))]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                color=COLORS["teal"],
                linewidth=1.8,
            )
        )

    ax.text(
        0.5,
        0.92,
        "Research-to-simulation-to-optimization pipeline",
        ha="center",
        va="center",
        weight="bold",
        fontsize=12,
        color=COLORS["dark"],
    )
    save(fig, "fig1_method_schematic.png")


def intrinsic_results() -> None:
    results = load_json("results.json")
    generalization = load_json("generalization_data.json")
    waveform = load_json("waveform_data.json")

    iterations = [row["iteration"] for row in results]
    worst = [row["worst_case_reward"] for row in results]
    mean = [row["mean_reward"] for row in results]
    x0 = [row["x0"] for row in results]
    coupling = [row["coupling_a"] for row in results]
    best_idx = int(np.argmax(worst))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    ax = axes[0, 0]
    ax.plot(iterations, worst, marker="o", color=COLORS["red"], label="Worst-case")
    ax.plot(iterations, mean, marker="s", linestyle="--", color=COLORS["blue"], label="Mean")
    ax.scatter([best_idx], [worst[best_idx]], s=70, color=COLORS["green"], zorder=4)
    ax.axhline(worst[0], color="#94a3b8", linestyle=":", linewidth=1.2)
    ax.set_title("A. Intrinsic minimax optimization")
    ax.set_xlabel("LLM iteration")
    ax.set_ylabel("Reward (higher is better)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    scatter = ax.scatter(x0, coupling, c=worst, cmap="RdYlGn", s=70, edgecolor="#111827")
    for i, (x_val, k_val) in enumerate(zip(x0, coupling, strict=True)):
        ax.text(x_val + 0.015, k_val + 0.00025, str(i), fontsize=7)
    ax.set_title("B. Proposed parameter sets")
    ax.set_xlabel("Epileptogenicity $x_0$")
    ax.set_ylabel("Coupling $K$")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Worst-case reward")

    ax = axes[1, 0]
    labels = ["Train\nbaseline", "Train\noptimized", "Test\nbaseline", "Test\noptimized"]
    groups = [
        generalization["train_baseline"],
        generalization["train_optimized"],
        generalization["test_baseline"],
        generalization["test_optimized"],
    ]
    positions = np.arange(len(groups))
    means = [group["mean"] for group in groups]
    ax.bar(
        positions,
        means,
        color=[COLORS["gray"], COLORS["green"], COLORS["gray"], COLORS["green"]],
        alpha=0.85,
    )
    for idx, group in enumerate(groups):
        jitter = np.linspace(-0.11, 0.11, len(group["rewards"]))
        ax.scatter(np.full(len(group["rewards"]), idx) + jitter, group["rewards"], s=13, color="#0f172a", alpha=0.55)
        ax.plot([idx - 0.18, idx + 0.18], [group["worst"], group["worst"]], color=COLORS["red"], linewidth=1.6)
    ax.set_xticks(positions, labels)
    ax.set_title("C. Held-out virtual patients")
    ax.set_ylabel("Reward")

    ax = axes[1, 1]
    base = waveform["baseline"]
    opt = waveform["optimized"]
    base_x1 = np.array(base["x1"])
    opt_x1 = np.array(opt["x1"])
    base_mean = base_x1.mean(axis=1)
    opt_mean = opt_x1.mean(axis=1)
    base_sem = base_x1.std(axis=1) / np.sqrt(base_x1.shape[1])
    opt_sem = opt_x1.std(axis=1) / np.sqrt(opt_x1.shape[1])
    ax.plot(base["time"], base_mean, color=COLORS["red"], linewidth=1.2, label="Baseline")
    ax.fill_between(base["time"], base_mean - base_sem, base_mean + base_sem, color=COLORS["red"], alpha=0.16, linewidth=0)
    ax.plot(opt["time"], opt_mean, color=COLORS["blue"], linewidth=1.2, label="Optimized")
    ax.fill_between(opt["time"], opt_mean - opt_sem, opt_mean + opt_sem, color=COLORS["blue"], alpha=0.16, linewidth=0)
    ax.set_title("D. Example seizure activity trace")
    ax.set_xlabel("Simulation time")
    ax.set_ylabel("$x_1$")
    ax.legend(frameon=False)

    fig.tight_layout()
    save(fig, "fig2_intrinsic_results.png")


def clinical_and_comparison_results() -> None:
    clinical = load_json("results_v2.json")
    cohort = load_json("cohort_results_20.json")
    comparison = load_json("bo_comparison.json")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))

    ax = axes[0, 0]
    hist = clinical["history"]
    iters = [row["iteration"] for row in hist]
    worst = [row["worst_case_reward"] for row in hist]
    mean = [row["mean_reward"] for row in hist]
    ax.plot(iters, worst, marker="o", color=COLORS["orange"], label="Worst-case")
    ax.plot(iters, mean, marker="s", linestyle="--", color=COLORS["blue"], label="Mean")
    ax.axhline(clinical["baseline_worst"], color="#94a3b8", linestyle=":", label="No stimulation")
    ax.set_title("A. External-stimulation search")
    ax.set_xlabel("LLM iteration")
    ax.set_ylabel("Reward")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    x = np.arange(len(comparison["bo_trajectory"]))
    ax.plot(x, comparison["bo_trajectory"], marker="o", color=COLORS["gray"], label="Bayesian optimization")
    ax.plot(x, comparison["llm_trajectory"], marker="o", color=COLORS["purple"], label="LLM-guided")
    ax.set_title("B. Knowledge-guided search comparison")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Worst-case reward")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    baseline = np.array(cohort["baseline"]["rewards"])
    optimized = np.array(cohort["optimized"]["rewards"])
    for b_val, o_val in zip(baseline, optimized, strict=True):
        color = COLORS["green"] if o_val > b_val else COLORS["red"]
        ax.plot([0, 1], [b_val, o_val], color=color, alpha=0.45, linewidth=1.0)
        ax.scatter([0, 1], [b_val, o_val], color=color, s=14)
    ax.set_xticks([0, 1], ["Baseline", "rHC protocol"])
    ax.set_ylabel("Reward")
    ax.set_title("C. Paired 20-patient cohort")
    ax.text(
        0.03,
        0.04,
        "Responder rate 55%; paradoxical worsening 45%",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#475569",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )

    ax = axes[1, 1]
    improvements = optimized - baseline
    soz = np.array(cohort["soz_types"])
    labels = ["temporal", "hippocampal", "frontal", "occipital"]
    y = np.arange(len(labels))
    means = []
    sems = []
    counts = []
    for label in labels:
        values = improvements[soz == label]
        means.append(float(np.mean(values)))
        sems.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
        counts.append(len(values))
    colors = [COLORS["green"] if value > 0 else COLORS["red"] for value in means]
    ax.axvline(0, color="#475569", linewidth=1.0)
    ax.errorbar(means, y, xerr=sems, fmt="none", ecolor="#334155", capsize=3, linewidth=1.1)
    ax.scatter(means, y, s=70, color=colors, zorder=3)
    ax.set_yticks(y, [f"{label} (n={count})" for label, count in zip(labels, counts, strict=True)])
    ax.set_xlabel("Mean reward change")
    ax.set_title("D. SOZ-stratified effect")

    fig.tight_layout()
    save(fig, "fig3_clinical_results.png")


def robustness_and_topology() -> None:
    generalization = load_json("generalization_data.json")
    topology = load_json("network_topology.json")
    rag = load_json("rag_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))

    ax = axes[0, 0]
    regimes = ["narrow", "medium", "wide"]
    base_worst = [generalization[f"{name}_baseline"]["worst"] for name in regimes]
    opt_worst = [generalization[f"{name}_optimized"]["worst"] for name in regimes]
    x = np.arange(len(regimes))
    width = 0.35
    ax.bar(x - width / 2, base_worst, width, label="Baseline", color=COLORS["gray"])
    ax.bar(x + width / 2, opt_worst, width, label="Optimized", color=COLORS["green"])
    ax.set_xticks(x, ["+/-0.003", "+/-0.005", "+/-0.010"])
    ax.set_title("A. Robustness across coupling variability")
    ax.set_xlabel("Perturbation regime")
    ax.set_ylabel("Worst-case reward")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    data = [
        generalization["stress_baseline"]["rewards"],
        generalization["stress_optimized"]["rewards"],
    ]
    parts = ax.violinplot(data, showmeans=True, showextrema=True)
    for body, color in zip(parts["bodies"], [COLORS["gray"], COLORS["green"]], strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor("#111827")
    ax.set_xticks([1, 2], ["Baseline", "Optimized"])
    ax.set_title("B. Stress test distribution (n=30)")
    ax.set_ylabel("Reward")

    ax = axes[1, 0]
    labels = ["rHC\n(target)", "rFEF\n(hub)", "rPFCORB\ntop hub"]
    strengths = [
        topology["rHC"]["strength"],
        topology["rFEF"]["strength"],
        topology["top10_hubs"][0]["strength"],
    ]
    degrees = [
        topology["rHC"]["degree"],
        topology["rFEF"]["degree"],
        topology["top10_hubs"][0]["degree"],
    ]
    ax.bar(np.arange(3) - 0.16, strengths, width=0.32, color=COLORS["blue"], label="Strength")
    ax.bar(np.arange(3) + 0.16, degrees, width=0.32, color=COLORS["teal"], label="Degree")
    ax.set_xticks(np.arange(3), labels)
    ax.set_title("C. Stimulation target in network context")
    ax.set_ylabel("Graph metric")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    hist = rag["history"]
    ax.plot([row["iteration"] for row in hist], [row["reward"] for row in hist], marker="o", color=COLORS["purple"])
    ax.axhline(rag["baseline"], color="#94a3b8", linestyle=":", label="Baseline")
    ax.set_title("D. RAG-augmented exploratory search")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Worst-case reward")
    ax.legend(frameon=False)

    fig.tight_layout()
    save(fig, "fig4_robustness_topology.png")


def main() -> None:
    setup_style()
    method_schematic()
    intrinsic_results()
    clinical_and_comparison_results()
    robustness_and_topology()
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
