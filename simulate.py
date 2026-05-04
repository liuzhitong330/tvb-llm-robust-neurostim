#!/usr/bin/env python3
"""Compatibility wrapper for intrinsic TVB simulations."""

from tvb_llm_neurostim.simulation import print_intrinsic_smoke_test, run_robust, run_simulation

__all__ = ["run_robust", "run_simulation"]


if __name__ == "__main__":
    print_intrinsic_smoke_test()
