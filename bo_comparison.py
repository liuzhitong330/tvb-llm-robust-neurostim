#!/usr/bin/env python3
"""Compare Bayesian optimization with the LLM optimizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from tvb_llm_neurostim.optimization import run_bo_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", type=Path, default=Path("bo_comparison.json"))
    parser.add_argument("--calls", type=int, default=8)
    args = parser.parse_args()
    run_bo_comparison(output_json=args.output, api_key=args.api_key, n_calls=args.calls)


if __name__ == "__main__":
    main()
