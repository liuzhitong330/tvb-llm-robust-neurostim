#!/usr/bin/env python3
"""Compatibility wrapper for clinical-style TVB simulations."""

from tvb_llm_neurostim.simulation import (
    get_clinical_recommendation,
    get_labels,
    print_clinical_smoke_test,
    run_robust_clinical,
    run_simulation_clinical,
)

__all__ = [
    "get_clinical_recommendation",
    "get_labels",
    "run_robust_clinical",
    "run_simulation_clinical",
]


if __name__ == "__main__":
    print_clinical_smoke_test()
