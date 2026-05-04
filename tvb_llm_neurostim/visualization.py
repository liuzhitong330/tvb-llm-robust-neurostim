"""Visualization utilities for checked-in optimization results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from tvb_llm_neurostim.config import PathsConfig


def render_results_plot(input_json: Path, output_png: Path) -> None:
    with input_json.open(encoding="utf-8") as handle:
        history = json.load(handle)

    iterations = [row["iteration"] for row in history]
    worst = [row["worst_case_reward"] for row in history]
    mean = [row["mean_reward"] for row in history]
    x0_values = [row["x0"] for row in history]
    couplings = [row["coupling_a"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(iterations, worst, "o-", color="crimson", label="Worst-case reward", linewidth=2)
    axes[0].plot(iterations, mean, "s--", color="steelblue", label="Mean reward", linewidth=2)
    axes[0].axhline(y=worst[0], color="gray", linestyle=":", label="Baseline worst-case")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Robust Optimization: Reward over Iterations")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    scatter = axes[1].scatter(
        x0_values,
        couplings,
        c=worst,
        cmap="RdYlGn",
        s=150,
        zorder=5,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.colorbar(scatter, ax=axes[1], label="Worst-case reward")
    axes[1].set_xlabel("x0 (epileptogenicity)")
    axes[1].set_ylabel("coupling_a")
    axes[1].set_title("Parameter Space Exploration")
    for index, row in enumerate(history):
        axes[1].annotate(
            str(index),
            (row["x0"], row["coupling_a"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render results.png from results.json.")
    parser.add_argument("--input", type=Path, default=PathsConfig().results_json)
    parser.add_argument("--output", type=Path, default=PathsConfig().results_png)
    args = parser.parse_args()
    render_results_plot(args.input, args.output)
