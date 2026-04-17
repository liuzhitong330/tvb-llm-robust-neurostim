#!/usr/bin/env python3
"""Run multiple PubMed queries, deduplicate by PMID, and save to a single CSV."""

import csv
import os
import time
from fetch_papers import search_pubmed, fetch_details, parse_article
from Bio import Entrez

EMAIL = "cathyliu014@gmail.com"
MAX_PER_QUERY = 100
OUTPUT_CSV = "all_papers.csv"

QUERIES = [
    '"virtual epileptic patient"',
    '"TVB" AND epilepsy AND stimulation',
    'epilepsy AND "brain stimulation" AND "computational model"',
    '"seizure suppression" AND "personalized model"',
    'epilepsy AND "digital twin"',
    'epilepsy AND "whole-brain model" AND treatment',
    '"seizure onset zone" AND "computational" AND stimulation',
    'epilepsy AND "network model" AND "brain stimulation"',
    '"responsive neurostimulation" AND computational',
    'epilepsy AND "in silico" AND stimulation',
]


def main():
    Entrez.email = EMAIL
    seen_pmids = set()
    all_papers = []

    # Load existing CSV to preserve prior results and skip known PMIDs
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_pmids.add(row["pmid"])
                all_papers.append(row)
        print(f"Loaded {len(all_papers)} existing papers from {OUTPUT_CSV}")

    for i, query in enumerate(QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(QUERIES)}: {query}")
        print("=" * 80)

        ids = search_pubmed(query, MAX_PER_QUERY)
        if not ids:
            print("  No results.")
            continue

        # Filter out already-seen PMIDs before fetching
        new_ids = [pid for pid in ids if pid not in seen_pmids]
        dupes = len(ids) - len(new_ids)
        if dupes:
            print(f"  Skipping {dupes} duplicate(s) already fetched.")

        if not new_ids:
            print("  All results are duplicates, skipping fetch.")
            continue

        print(f"  Fetching {len(new_ids)} new article(s)...")
        articles = fetch_details(new_ids)
        for article in articles:
            paper = parse_article(article)
            paper["query"] = query
            if paper["pmid"] not in seen_pmids:
                seen_pmids.add(paper["pmid"])
                all_papers.append(paper)
                print(f"  + [{paper['pmid']}] {paper['title'][:90]}")

        # Be polite to NCBI: pause between queries
        if i < len(QUERIES):
            time.sleep(1)

    # Sort by date descending
    all_papers.sort(key=lambda p: p["date"], reverse=True)

    # Write CSV
    if all_papers:
        fieldnames = ["pmid", "title", "authors", "journal", "date", "doi", "abstract", "url", "query"]
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_papers)

    print(f"\n{'='*80}")
    print(f"DONE: {len(all_papers)} unique papers saved to {OUTPUT_CSV}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
