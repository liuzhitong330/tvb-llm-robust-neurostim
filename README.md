# LLM-Guided Robust Optimization for Epilepsy Neurostimulation

Can a model-guided search find interventions that remain useful across different simulated neural systems—not just one favorable case?

This exploratory project combines The Virtual Brain's Epileptor model with an LLM-guided parameter search. It examines intrinsic model parameters and a separate external-input experiment, using a worst-case objective to test robustness across sampled model variation.

[Read the research essay](https://liuzhitong330.github.io/tvb-llm-robust-neurostim/) · [Paper PDF](https://liuzhitong330.github.io/tvb-llm-robust-neurostim/paper/tvb_llm_robust_neurostimulation.pdf) · [Code](src/tvb_llm_neurostim/) · [Run locally](#setup)

## What the Results Show—and Do Not Show

In the checked-in intrinsic-parameter experiment, the best worst-case reward changed from −0.5285 to −0.3182: a 39.8% improvement relative to the baseline reward magnitude. This reward is a simulated activity-variance proxy, not a measured reduction in clinical seizures.

The separate 20-model external-input evaluation illustrates the generalization problem: the proxy improved in 11 models and worsened in 9, with no overall improvement in the cohort mean. These results motivate examining individual responses rather than treating one optimized setting as universally effective.

This is an in-silico proof of concept using sampled model variation, not patient-specific clinical validation or a treatment recommendation. LLM-generated literature summaries and search rationales are hypotheses to check, not evidence of safety or efficacy.

## Explore the Project

- **Research narrative:** [interactive essay](https://liuzhitong330.github.io/tvb-llm-robust-neurostim/) and [paper](paper/).
- **Implementation and evidence:** [Python package](src/tvb_llm_neurostim/), [research scripts](scripts/), and [checked-in result artifacts](results/).
- **Reproduction and checks:** [setup](#setup), [commands](#common-commands), and [tests](tests/). Tests check code behavior and stored artifacts; they do not independently reproduce the expensive simulation or API runs.
- **Literature-mining provenance:** [methodology appendix](METHODOLOGY.md), including prompts and limitations.

## What Is In This Repository

- `index.html` and `brain3d.html` power the GitHub Pages essay.
- `src/tvb_llm_neurostim/` contains the package implementation.
- `scripts/` contains runnable research scripts and legacy entrypoints.
- `results/` contains checked-in JSON and plot artifacts used by the paper and site.
- `paper/` contains the LaTeX manuscript, generated figures, and PDF.
- `tests/` contains integration tests over the package, result artifacts, and static site.

## Pipeline

1. Literature mining: PubMed retrieval, LLM gap extraction, and idea ranking.
2. TVB simulations: Epileptor models with sampled parameter variation, not patient-fitted clinical digital twins.
3. Robust optimization: Claude proposes parameters, TVB evaluates them, and the objective maximizes worst-case reward.

The public essay deliberately starts with the lay explanation first, then moves into the technical details, charts, and limitations.

## Setup

Use `uv` for all Python package management and execution.

```bash
uv sync
```

Optional dependencies:

```bash
uv sync --extra analysis
uv sync --extra tvb
```

The `tvb` extra is needed for actual TVB simulations. Tests and static artifact checks do not rerun expensive TVB or API jobs.

## Common Commands

```bash
uv run pytest
uv run ruff check .
uv run tvb-visualize --input results/results.json --output results/results.png
uv run tvb-fetch-corpus --email you@example.com
uv run tvb-extract-gaps --api-key "$ANTHROPIC_API_KEY"
uv run tvb-rank-ideas --api-key "$ANTHROPIC_API_KEY"
uv run python scripts/rl_loop.py --api-key "$ANTHROPIC_API_KEY"
uv run python scripts/rl_loop_v2.py --api-key "$ANTHROPIC_API_KEY"
```

Do not use `pip`, `poetry`, `conda`, or manual virtualenv activation for this project.

## Result Artifacts

The repository keeps the published results as static artifacts:

- `results/results.json` and `results/results.png`: intrinsic minimax optimization trajectory.
- `results/results_v2.json`: clinical-style external-stimulation optimization.
- `results/cohort_results_20.json`: evaluation across 20 simulated models, not a clinical patient cohort.
- `results/bo_comparison.json`: LLM versus Bayesian optimization comparison.
- `results/clinical_landscape.json`: all-region external-stimulation landscape and random-search baselines.
- `results/rag_results.json`: RAG-augmented optimizer trace.
- `results/brain3d_data.json`, `results/waveform_data.json`, `results/generalization_data.json`, and related JSON files: website visualizations and paper figures.

The cleanup does not regenerate or replace these results by default.

## GitHub Pages

The site is static and targets GitHub Pages. Open `index.html` directly or serve the repository root with any static server. The `.nojekyll` file is intentionally present.

## Caveats

This is a proof-of-concept research pipeline, not clinical guidance. The literature-mining outputs are LLM-generated and should be treated as hypothesis generation. The TVB simulations model a sampled virtual cohort and do not establish real-world patient efficacy without clinical validation.

## Citation

```bibtex
@software{tvb_llm_neurostim_2026,
  title  = {LLM-Guided Robust Optimization for Epilepsy Neurostimulation},
  author = {Cathy Liu},
  year   = {2026},
  url    = {https://github.com/liuzhitong330/tvb-llm-robust-neurostim},
  note   = {Built on The Virtual Brain platform}
}
```
