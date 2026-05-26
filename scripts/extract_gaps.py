#!/usr/bin/env python3
"""Compatibility wrapper for Claude-based gap extraction."""

from tvb_llm_neurostim.literature import extract_gaps, main_extract_gaps

__all__ = ["extract_gaps"]


if __name__ == "__main__":
    main_extract_gaps()
