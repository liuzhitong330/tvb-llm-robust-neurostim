#!/usr/bin/env python3
"""Compatibility wrapper for research-idea ranking."""

from tvb_llm_neurostim.ranking import collect_ideas, main_rank_ideas, parse_json_response

__all__ = ["collect_ideas", "parse_json_response"]


if __name__ == "__main__":
    main_rank_ideas()
