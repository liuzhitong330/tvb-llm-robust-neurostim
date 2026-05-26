#!/usr/bin/env python3
"""Run the clinical-style LLM minimax optimization loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from tvb_llm_neurostim.config import PathsConfig
from tvb_llm_neurostim.optimization import run_clinical_optimization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output", type=Path, default=PathsConfig().clinical_results_json)
    args = parser.parse_args()
    run_clinical_optimization(output_json=args.output, api_key=args.api_key)


if __name__ == "__main__":
    main()
