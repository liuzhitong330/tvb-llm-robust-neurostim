#!/usr/bin/env python3
"""Fetch PubMed papers about virtual brain epilepsy stimulation using Biopython Entrez."""

import argparse
import csv
import sys
from Bio import Entrez


def search_pubmed(query, max_results=50):
    """Search PubMed and return a list of article IDs."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    results = Entrez.read(handle)
    handle.close()
    print(f"Found {results['Count']} total results, fetching top {len(results['IdList'])}.")
    return results["IdList"]


def fetch_details(id_list):
    """Fetch article details for a list of PubMed IDs."""
    if not id_list:
        return []
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="xml")
    records = Entrez.read(handle)
    handle.close()
    return records["PubmedArticle"]


def parse_article(article):
    """Extract key fields from a PubMed article record."""
    medline = article["MedlineCitation"]
    art = medline["Article"]
    pmid = str(medline["PMID"])
    title = art.get("ArticleTitle", "N/A")

    # Authors
    authors = []
    if "AuthorList" in art:
        for author in art["AuthorList"]:
            last = author.get("LastName", "")
            first = author.get("ForeName", "")
            if last:
                authors.append(f"{last} {first}".strip())
    author_str = "; ".join(authors) if authors else "N/A"

    # Abstract
    abstract = ""
    if "Abstract" in art and "AbstractText" in art["Abstract"]:
        abstract = " ".join(str(t) for t in art["Abstract"]["AbstractText"])

    # Date
    date_parts = art.get("ArticleDate", [])
    if date_parts:
        d = date_parts[0]
        date_str = f"{d.get('Year', '')}-{d.get('Month', '').zfill(2)}-{d.get('Day', '').zfill(2)}"
    else:
        pd = medline.get("DateCompleted", medline.get("DateRevised", {}))
        date_str = pd.get("Year", "N/A") if pd else "N/A"

    # Journal
    journal = art.get("Journal", {}).get("Title", "N/A")

    # DOI
    doi = ""
    for eid in art.get("ELocationID", []):
        if eid.attributes.get("EIdType") == "doi":
            doi = str(eid)
            break

    return {
        "pmid": pmid,
        "title": title,
        "authors": author_str,
        "journal": journal,
        "date": date_str,
        "doi": doi,
        "abstract": abstract,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="Email for NCBI Entrez (required by NCBI)")
    parser.add_argument("--query", default='"virtual brain" AND epilepsy AND stimulation',
                        help="PubMed search query")
    parser.add_argument("--max", type=int, default=50, help="Max results to fetch (default: 50)")
    parser.add_argument("--csv", dest="csv_file", help="Save results to CSV file")
    args = parser.parse_args()

    Entrez.email = args.email

    print(f"Searching PubMed: {args.query}")
    ids = search_pubmed(args.query, args.max)
    if not ids:
        print("No results found.")
        sys.exit(0)

    print("Fetching article details...")
    articles = fetch_details(ids)
    papers = [parse_article(a) for a in articles]

    # Print to console
    for i, p in enumerate(papers, 1):
        print(f"\n{'='*80}")
        print(f"[{i}] {p['title']}")
        print(f"    Authors:  {p['authors']}")
        print(f"    Journal:  {p['journal']} ({p['date']})")
        print(f"    DOI:      {p['doi'] or 'N/A'}")
        print(f"    URL:      {p['url']}")
        if p["abstract"]:
            print(f"    Abstract: {p['abstract'][:200]}...")

    # Save CSV if requested
    if args.csv_file:
        with open(args.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=papers[0].keys())
            writer.writeheader()
            writer.writerows(papers)
        print(f"\nSaved {len(papers)} papers to {args.csv_file}")

    print(f"\nTotal: {len(papers)} papers fetched.")


if __name__ == "__main__":
    main()
