# LLM-Guided Robust Optimization for Epilepsy Neurostimulation

Public research essay and reproducible code for Cathy Liu's TVB + LLM neurostimulation project.

The project asks whether a large language model can help search for brain-stimulation parameters that are robust across virtual epilepsy patients. The core result: in the intrinsic-parameter experiment, the best protocol improved worst-case reward by 39.8% over baseline while preserving the existing result artifacts used by the GitHub Pages site.

## What Is In This Repository

- `index.html`, `brain3d.html`, and checked-in JSON artifacts power the GitHub Pages essay.
- `tvb_llm_neurostim/` contains the package implementation.
- Root Python files such as `simulate.py`, `rl_loop.py`, and `rank_ideas.py` are compatibility wrappers so existing commands still work.
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
uv run python visualize.py --input results.json --output results.png
uv run python fetch_all_papers.py --email you@example.com
uv run python extract_gaps.py --api-key "$ANTHROPIC_API_KEY"
uv run python rank_ideas.py --api-key "$ANTHROPIC_API_KEY"
uv run python rl_loop.py --api-key "$ANTHROPIC_API_KEY"
uv run python rl_loop_v2.py --api-key "$ANTHROPIC_API_KEY"
```

Do not use `pip`, `poetry`, `conda`, or manual virtualenv activation for this project.

## Result Artifacts

The repository keeps the published results as static artifacts:

- `results.json` and `results.png`: intrinsic minimax optimization trajectory.
- `results_v2.json`: clinical-style external-stimulation optimization.
- `cohort_results_20.json`: 20-patient cohort study.
- `bo_comparison.json`: LLM versus Bayesian optimization comparison.
- `rag_results.json`: RAG-augmented optimizer trace.
- `brain3d_data.json`, `waveform_data.json`, `generalization_data.json`, and related JSON files: website visualizations.

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
