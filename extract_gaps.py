#!/usr/bin/env python3
"""Extract research gaps from papers in all_papers.csv using Claude API."""

import argparse
import csv
import json
import time
import anthropic

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a neuroscience research analyst. Given a paper's title and abstract, extract structured information about research gaps. Return valid JSON only, no markdown fences. Use this exact schema:

{
  "open_questions": ["list of specific unanswered research questions mentioned or implied"],
  "future_experiments": ["concrete experiments or studies suggested by the authors"],
  "limitations": ["methodological limitations explicitly stated"],
  "tvb_relevant": true/false
}

Rules:
- open_questions: Extract specific scientific questions that remain unanswered. Be concrete, not vague.
- future_experiments: Only include experiments/studies the authors explicitly suggest or that directly follow from stated limitations.
- limitations: Only include limitations the authors explicitly acknowledge.
- tvb_relevant: true if the paper is relevant to The Virtual Brain platform, whole-brain computational modeling, or patient-specific brain network simulations. false otherwise.
- If a field has no items, use an empty list [].
- Return ONLY the JSON object, nothing else."""


def extract_gaps(client, title, abstract):
    """Send title+abstract to Claude and parse structured JSON response."""
    user_msg = f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}\n\nAbstract: [not available]"

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "")
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text), response.usage
    except json.JSONDecodeError:
        return {"open_questions": [], "future_experiments": [], "limitations": [], "tvb_relevant": False, "_parse_error": text}, response.usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="all_papers.csv", help="Input CSV (default: all_papers.csv)")
    parser.add_argument("--output", default="gaps.json", help="Output JSON (default: gaps.json)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N papers (for testing)")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()

    # Load papers
    with open(args.input, newline="", encoding="utf-8") as f:
        papers = list(csv.DictReader(f))

    if args.limit:
        papers = papers[:args.limit]

    print(f"Processing {len(papers)} papers with {MODEL}...")

    client = anthropic.Anthropic(api_key=args.api_key) if args.api_key else anthropic.Anthropic()
    results = []
    total_input = 0
    total_output = 0

    for i, paper in enumerate(papers, 1):
        pmid = paper["pmid"]
        title = paper["title"]
        abstract = paper.get("abstract", "")

        print(f"  [{i}/{len(papers)}] PMID {pmid}: {title[:70]}...")

        gaps, usage = extract_gaps(client, title, abstract)
        total_input += usage.input_tokens
        total_output += usage.output_tokens

        results.append({
            "pmid": pmid,
            "title": title,
            "journal": paper.get("journal", ""),
            "date": paper.get("date", ""),
            "url": paper.get("url", ""),
            "gaps": gaps,
        })

        # Rate limiting: small pause between requests
        if i < len(papers):
            time.sleep(0.5)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary stats
    tvb_count = sum(1 for r in results if r["gaps"].get("tvb_relevant"))
    total_questions = sum(len(r["gaps"].get("open_questions", [])) for r in results)
    total_experiments = sum(len(r["gaps"].get("future_experiments", [])) for r in results)
    total_limitations = sum(len(r["gaps"].get("limitations", [])) for r in results)
    parse_errors = sum(1 for r in results if "_parse_error" in r["gaps"])

    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")
    print(f"Papers processed:    {len(results)}")
    print(f"TVB-relevant:        {tvb_count}/{len(results)}")
    print(f"Open questions:      {total_questions}")
    print(f"Future experiments:  {total_experiments}")
    print(f"Limitations:         {total_limitations}")
    print(f"Parse errors:        {parse_errors}")
    print(f"Tokens used:         {total_input:,} input / {total_output:,} output")
    est_cost = (total_input / 1_000_000 * 1.0) + (total_output / 1_000_000 * 5.0)
    print(f"Estimated cost:      ${est_cost:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
