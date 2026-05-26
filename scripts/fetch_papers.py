#!/usr/bin/env python3
"""Compatibility wrapper for one-off PubMed retrieval."""

from tvb_llm_neurostim.pubmed import fetch_details, main_fetch_papers, parse_article, search_pubmed

__all__ = ["fetch_details", "parse_article", "search_pubmed"]


if __name__ == "__main__":
    main_fetch_papers()
