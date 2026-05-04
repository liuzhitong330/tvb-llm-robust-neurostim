"""PubMed retrieval utilities."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from Bio import Entrez
from pydantic import BaseModel, ConfigDict

from tvb_llm_neurostim.config import LiteratureMiningConfig, PathsConfig

DEFAULT_LITERATURE = LiteratureMiningConfig()
DEFAULT_PATHS = PathsConfig()


class Paper(BaseModel):
    """Normalized PubMed article metadata used throughout the pipeline."""

    model_config = ConfigDict(extra="forbid")

    pmid: str
    title: str
    authors: str = ""
    journal: str = ""
    date: str = ""
    doi: str = ""
    abstract: str = ""
    url: str = ""
    query: str = ""

    def csv_row(self) -> dict[str, str]:
        return self.model_dump()


PAPER_FIELDNAMES = [
    "pmid",
    "title",
    "authors",
    "journal",
    "date",
    "doi",
    "abstract",
    "url",
    "query",
]


def search_pubmed(query: str, max_results: int = 50) -> list[str]:
    """Search PubMed and return article IDs ordered by relevance."""

    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    try:
        results = Entrez.read(handle)
    finally:
        handle.close()
    print(f"Found {results['Count']} total results, fetching top {len(results['IdList'])}.")
    return list(results["IdList"])


def fetch_details(id_list: list[str]) -> list[Any]:
    """Fetch PubMed article details for a list of PMIDs."""

    if not id_list:
        return []
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="xml")
    try:
        records = Entrez.read(handle)
    finally:
        handle.close()
    return list(records["PubmedArticle"])


def parse_article(article: Any, *, query: str = "") -> Paper:
    """Extract stable metadata from a Bio.Entrez PubMedArticle record."""

    medline = article["MedlineCitation"]
    article_data = medline["Article"]
    pmid = str(medline["PMID"])
    title = str(article_data.get("ArticleTitle", "N/A"))

    authors = []
    for author in article_data.get("AuthorList", []):
        last = author.get("LastName", "")
        first = author.get("ForeName", "")
        if last:
            authors.append(f"{last} {first}".strip())

    abstract = ""
    if "Abstract" in article_data and "AbstractText" in article_data["Abstract"]:
        abstract = " ".join(str(part) for part in article_data["Abstract"]["AbstractText"])

    article_dates = article_data.get("ArticleDate", [])
    if article_dates:
        date = article_dates[0]
        date_str = (
            f"{date.get('Year', '')}-"
            f"{date.get('Month', '').zfill(2)}-"
            f"{date.get('Day', '').zfill(2)}"
        )
    else:
        pub_date = medline.get("DateCompleted", medline.get("DateRevised", {}))
        date_str = pub_date.get("Year", "N/A") if pub_date else "N/A"

    doi = ""
    for element_id in article_data.get("ELocationID", []):
        if element_id.attributes.get("EIdType") == "doi":
            doi = str(element_id)
            break

    return Paper(
        pmid=pmid,
        title=title,
        authors="; ".join(authors) if authors else "N/A",
        journal=article_data.get("Journal", {}).get("Title", "N/A"),
        date=date_str,
        doi=doi,
        abstract=abstract,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        query=query,
    )


def read_papers_csv(path: Path) -> list[Paper]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [Paper(**row) for row in csv.DictReader(handle)]


def write_papers_csv(path: Path, papers: list[Paper]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_FIELDNAMES)
        writer.writeheader()
        for paper in papers:
            writer.writerow(paper.csv_row())


def run_query(query: str, *, email: str, max_results: int) -> list[Paper]:
    Entrez.email = email
    ids = search_pubmed(query, max_results)
    articles = fetch_details(ids)
    return [parse_article(article, query=query) for article in articles]


def run_corpus_fetch(
    *,
    config: LiteratureMiningConfig = DEFAULT_LITERATURE,
    output_csv: Path = DEFAULT_PATHS.papers_csv,
) -> list[Paper]:
    """Run the configured PubMed corpus retrieval and write a deduplicated CSV."""

    Entrez.email = config.email
    papers = read_papers_csv(output_csv)
    seen_pmids = {paper.pmid for paper in papers}
    if papers:
        print(f"Loaded {len(papers)} existing papers from {output_csv}")

    for index, query in enumerate(config.queries, start=1):
        print(f"\n{'=' * 80}")
        print(f"Query {index}/{len(config.queries)}: {query}")
        print("=" * 80)

        ids = search_pubmed(query, config.max_per_query)
        new_ids = [pmid for pmid in ids if pmid not in seen_pmids]
        if len(new_ids) < len(ids):
            print(f"  Skipping {len(ids) - len(new_ids)} duplicate(s) already fetched.")
        if not new_ids:
            print("  All results are duplicates, skipping fetch.")
            continue

        print(f"  Fetching {len(new_ids)} new article(s)...")
        for article in fetch_details(new_ids):
            paper = parse_article(article, query=query)
            if paper.pmid in seen_pmids:
                continue
            seen_pmids.add(paper.pmid)
            papers.append(paper)
            print(f"  + [{paper.pmid}] {paper.title[:90]}")

        if index < len(config.queries):
            time.sleep(config.request_pause_seconds)

    papers.sort(key=lambda paper: paper.date, reverse=True)
    write_papers_csv(output_csv, papers)
    print(f"\nDONE: {len(papers)} unique papers saved to {output_csv}")
    return papers


def main_fetch_papers() -> None:
    parser = argparse.ArgumentParser(description="Fetch one PubMed query to console or CSV.")
    parser.add_argument("--email", required=True, help="Email for NCBI Entrez.")
    parser.add_argument(
        "--query",
        default='"virtual brain" AND epilepsy AND stimulation',
        help="PubMed search query.",
    )
    parser.add_argument("--max", type=int, default=50, help="Max results to fetch.")
    parser.add_argument("--csv", dest="csv_file", help="Optional CSV output path.")
    args = parser.parse_args()

    papers = run_query(args.query, email=args.email, max_results=args.max)
    for index, paper in enumerate(papers, start=1):
        print(f"\n{'=' * 80}")
        print(f"[{index}] {paper.title}")
        print(f"    Authors:  {paper.authors}")
        print(f"    Journal:  {paper.journal} ({paper.date})")
        print(f"    DOI:      {paper.doi or 'N/A'}")
        print(f"    URL:      {paper.url}")
        if paper.abstract:
            print(f"    Abstract: {paper.abstract[:200]}...")

    if args.csv_file:
        write_papers_csv(Path(args.csv_file), papers)
        print(f"\nSaved {len(papers)} papers to {args.csv_file}")
    print(f"\nTotal: {len(papers)} papers fetched.")


def main_fetch_all_papers() -> None:
    parser = argparse.ArgumentParser(description="Fetch the configured PubMed corpus.")
    parser.add_argument("--email", default=DEFAULT_LITERATURE.email)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATHS.papers_csv)
    parser.add_argument("--max-per-query", type=int, default=DEFAULT_LITERATURE.max_per_query)
    args = parser.parse_args()

    config = LiteratureMiningConfig(email=args.email, max_per_query=args.max_per_query)
    run_corpus_fetch(config=config, output_csv=args.output)
