# Academic Paper Artifact

This folder contains the arXiv-style manuscript for Cathy Liu's TVB + LLM neurostimulation project.

## Files

- `main.tex`: formal academic manuscript.
- `arxiv.sty`: small local style shim so `\usepackage{arxiv}` works without downloading a template.
- `generate_figures.py`: regenerates manuscript plots from checked-in JSON artifacts.
- `images/`: generated manuscript figures.

## Rebuild Figures

Use `uv` from the repository root:

```bash
uv run python paper/generate_figures.py
```

## Compile Manuscript

If a TeX distribution is installed:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

The manuscript intentionally reports only the existing repository artifacts; it does not create new simulation results.
