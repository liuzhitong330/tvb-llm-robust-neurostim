"""LLM-based extraction of research gaps from PubMed metadata."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, ConfigDict, Field

from tvb_llm_neurostim.config import LiteratureMiningConfig, ModelConfig, PathsConfig
from tvb_llm_neurostim.json_utils import parse_json_response

DEFAULT_LITERATURE = LiteratureMiningConfig()
DEFAULT_MODELS = ModelConfig()
DEFAULT_PATHS = PathsConfig()


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


class ExtractedGaps(BaseModel):
    model_config = ConfigDict(extra="allow")

    open_questions: list[str] = Field(default_factory=list)
    future_experiments: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    tvb_relevant: bool = False


class PaperGapResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pmid: str
    title: str
    journal: str = ""
    date: str = ""
    url: str = ""
    gaps: ExtractedGaps


def _text_from_anthropic_response(response: Any) -> str:
    return next((block.text for block in response.content if block.type == "text"), "")


def extract_gaps(
    client: anthropic.Anthropic,
    title: str,
    abstract: str,
    *,
    model_config: ModelConfig = DEFAULT_MODELS,
) -> tuple[ExtractedGaps, Any]:
    """Send one paper title/abstract to Claude and parse the structured gaps."""

    user_message = (
        f"Title: {title}\n\nAbstract: {abstract}"
        if abstract
        else f"Title: {title}\n\nAbstract: [not available]"
    )
    response = client.messages.create(
        model=model_config.haiku_model,
        max_tokens=model_config.extraction_max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = _text_from_anthropic_response(response)
    try:
        return ExtractedGaps(**parse_json_response(text, context=f"gap extraction for {title}")), response.usage
    except ValueError:
        return (
            ExtractedGaps(
                open_questions=[],
                future_experiments=[],
                limitations=[],
                tvb_relevant=False,
                _parse_error=text,
            ),
            response.usage,
        )


def load_papers(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        papers = list(csv.DictReader(handle))
    return papers[:limit] if limit else papers


def run_gap_extraction(
    *,
    input_csv: Path = DEFAULT_PATHS.papers_csv,
    output_json: Path = DEFAULT_PATHS.gaps_json,
    limit: int | None = None,
    api_key: str | None = None,
    literature_config: LiteratureMiningConfig = DEFAULT_LITERATURE,
    model_config: ModelConfig = DEFAULT_MODELS,
) -> list[PaperGapResult]:
    """Extract gaps for papers in a CSV and write the JSON artifact."""

    papers = load_papers(input_csv, limit)
    print(f"Processing {len(papers)} papers with {model_config.haiku_model}...")
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    results: list[PaperGapResult] = []
    total_input = 0
    total_output = 0
    for index, paper in enumerate(papers, start=1):
        print(f"  [{index}/{len(papers)}] PMID {paper['pmid']}: {paper['title'][:70]}...")
        gaps, usage = extract_gaps(
            client,
            paper["title"],
            paper.get("abstract", ""),
            model_config=model_config,
        )
        total_input += usage.input_tokens
        total_output += usage.output_tokens
        results.append(
            PaperGapResult(
                pmid=paper["pmid"],
                title=paper["title"],
                journal=paper.get("journal", ""),
                date=paper.get("date", ""),
                url=paper.get("url", ""),
                gaps=gaps,
            )
        )
        if index < len(papers):
            time.sleep(literature_config.extraction_pause_seconds)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump([result.model_dump() for result in results], handle, indent=2, ensure_ascii=False)

    tvb_count = sum(result.gaps.tvb_relevant for result in results)
    total_questions = sum(len(result.gaps.open_questions) for result in results)
    total_experiments = sum(len(result.gaps.future_experiments) for result in results)
    total_limitations = sum(len(result.gaps.limitations) for result in results)
    parse_errors = sum("_parse_error" in result.gaps.model_extra for result in results)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_json}")
    print(f"Papers processed:    {len(results)}")
    print(f"TVB-relevant:        {tvb_count}/{len(results)}")
    print(f"Open questions:      {total_questions}")
    print(f"Future experiments:  {total_experiments}")
    print(f"Limitations:         {total_limitations}")
    print(f"Parse errors:        {parse_errors}")
    print(f"Tokens used:         {total_input:,} input / {total_output:,} output")
    estimated_cost = (total_input / 1_000_000 * 1.0) + (total_output / 1_000_000 * 5.0)
    print(f"Estimated cost:      ${estimated_cost:.4f}")
    print(f"{'=' * 60}")
    return results


def main_extract_gaps() -> None:
    parser = argparse.ArgumentParser(description="Extract structured research gaps with Claude.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PATHS.papers_csv)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATHS.gaps_json)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    run_gap_extraction(
        input_csv=args.input,
        output_json=args.output,
        limit=args.limit,
        api_key=args.api_key,
    )
