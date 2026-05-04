#!/usr/bin/env python3
"""Run the RAG-augmented stimulation optimizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from tvb_llm_neurostim.rag import rag_optimize, retrieve_knowledge

__all__ = ["rag_optimize", "retrieve_knowledge"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--gaps", type=Path, default=Path("gaps.json"))
    parser.add_argument("--output", type=Path, default=Path("rag_results.json"))
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    rag_optimize(
        gaps_json=args.gaps,
        output_json=args.output,
        api_key=args.api_key,
        n_iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
