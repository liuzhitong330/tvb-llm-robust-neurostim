# LLM-Guided Robust Optimization for Epilepsy Neurostimulation

Public research essay and reproducible code for Cathy Liu's TVB + LLM neurostimulation project.

The project asks whether a large language model can help search for brain-stimulation parameters that are robust across virtual epilepsy patients. The core result: in the intrinsic-parameter experiment, the best protocol improved worst-case reward by 39.8% over baseline while preserving the existing result artifacts used by the GitHub Pages site.

## What Is In This Repository

- `index.html` and `brain3d.html` power the GitHub Pages essay.
- `src/tvb_llm_neurostim/` contains the package implementation.
- `scripts/` contains runnable research scripts and legacy entrypoints.
- `results/` contains checked-in JSON and plot artifacts used by the paper and site.
- `paper/` contains the LaTeX manuscript, generated figures, and PDF.
- `tests/` contains integration tests over the package, result artifacts, and static site.

## Pipeline

1. Literature mining: PubMed retrieval, LLM gap extraction, and idea ranking.
2. TVB digital twins: Epileptor simulations across virtual patient variability.
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
- `results/cohort_results_20.json`: 20-patient cohort study.
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
