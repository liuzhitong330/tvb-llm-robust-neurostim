"""Cluster and rank literature-mined research ideas."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, ConfigDict, Field

from tvb_llm_neurostim.config import ModelConfig, PathsConfig
from tvb_llm_neurostim.json_utils import parse_json_response

DEFAULT_MODELS = ModelConfig()
DEFAULT_PATHS = PathsConfig()


GROUPING_SYSTEM = """You are a neuroscience research synthesizer specializing in computational brain modeling and epilepsy.

Given a list of raw research ideas (questions and experiments), group them into coherent thematic clusters.
Each cluster should represent a distinct, actionable research direction relevant to computational neuroscience or epilepsy treatment.

Return ONLY valid JSON -- no markdown fences, no commentary -- in this exact schema:
{
  "clusters": [
    {
      "theme": "short theme title (5-10 words)",
      "summary": "1-2 sentence synthesis of what this cluster is about",
      "representative_idea": "the single clearest, most concrete idea from this cluster",
      "idea_count": <integer>,
      "tvb_relevant": <true/false>,
      "source_pmids": ["pmid1", "pmid2", ...]
    }
  ]
}

Guidelines:
- Aim for 25-40 meaningful clusters (merge near-duplicates, split unrelated ideas)
- Prefer clusters with direct computational modeling or brain stimulation angles
- tvb_relevant = true if the cluster can be addressed with whole-brain network modeling
- source_pmids: list up to 5 most relevant PMIDs for that cluster
"""


JUDGE_SYSTEM = """You are a senior neuroscience grant reviewer with deep expertise in:
- Computational brain modeling (The Virtual Brain platform, connectome-based models)
- Drug-resistant epilepsy treatment (resection surgery, brain stimulation, closed-loop systems)
- Translational neuroscience (bench-to-bedside pipeline)

Score each research idea cluster on three dimensions (1-10 each):

NOVELTY (1-10):
  1-3: Well-established, being actively pursued, incremental
  4-6: Some novelty, gaps remain, moderate competition
  7-9: Clearly underexplored, fresh angle, high originality
  10: Genuinely unexplored territory, paradigm-shifting potential

TVB_FEASIBILITY (1-10): How readily can The Virtual Brain address this?
  1-3: Requires fundamentally different tools/data not in TVB
  4-6: Possible with TVB extensions or new data pipelines
  7-9: Directly addressable with current/near-term TVB capabilities
  10: TVB is the ideal or only tool for this

CLINICAL_IMPACT (1-10): Potential benefit to drug-resistant epilepsy patients
  1-3: Basic science, distant clinical translation
  4-6: Moderate path to clinical use within 5-10 years
  7-9: High impact, clear clinical application within 5 years
  10: Could immediately change treatment decisions

Return ONLY valid JSON -- no markdown, no commentary:
{
  "scores": [
    {
      "theme": "<exact theme string from input>",
      "novelty": <1-10>,
      "tvb_feasibility": <1-10>,
      "clinical_impact": <1-10>,
      "total": <sum>,
      "rationale": "2-3 sentence justification covering all three scores",
      "key_opportunity": "one concrete sentence describing the most promising next step"
    }
  ]
}
"""


class Source(BaseModel):
    model_config = ConfigDict(extra="allow")

    pmid: str
    title: str
    url: str = ""
    tvb_relevant: bool = False


class RawIdea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    type: str
    source: Source


class Cluster(BaseModel):
    model_config = ConfigDict(extra="allow")

    theme: str
    summary: str = ""
    representative_idea: str = ""
    idea_count: int = 0
    tvb_relevant: bool = False
    source_pmids: list[str] = Field(default_factory=list)


class ScoredCluster(Cluster):
    novelty: int
    tvb_feasibility: int
    clinical_impact: int
    total: int
    rationale: str = ""
    key_opportunity: str = ""


def collect_ideas(gaps_data: list[dict[str, Any]]) -> list[RawIdea]:
    ideas: list[RawIdea] = []
    for paper in gaps_data:
        source = Source(
            pmid=str(paper["pmid"]),
            title=paper["title"],
            url=paper.get("url", ""),
            tvb_relevant=paper.get("gaps", {}).get("tvb_relevant", False),
        )
        for question in paper.get("gaps", {}).get("open_questions", []):
            ideas.append(RawIdea(text=question, type="question", source=source))
        for experiment in paper.get("gaps", {}).get("future_experiments", []):
            ideas.append(RawIdea(text=experiment, type="experiment", source=source))
    return ideas


def group_ideas(
    client: anthropic.Anthropic,
    ideas: list[RawIdea],
    *,
    batch_size: int = 150,
    model_config: ModelConfig = DEFAULT_MODELS,
) -> list[Cluster]:
    print(f"\n[Grouping] {len(ideas)} raw ideas -> thematic clusters...")
    all_clusters: list[Cluster] = []
    batches = [ideas[index : index + batch_size] for index in range(0, len(ideas), batch_size)]

    for batch_index, batch in enumerate(batches, start=1):
        print(f"  Batch {batch_index}/{len(batches)} ({len(batch)} ideas)...")
        idea_lines = "\n".join(
            f"[{index + 1}] ({item.type}, PMID:{item.source.pmid}) {item.text}"
            for index, item in enumerate(batch)
        )
        response = client.messages.create(
            model=model_config.grouping_model,
            max_tokens=model_config.grouping_max_tokens,
            system=GROUPING_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Group these {len(batch)} research ideas:\n\n{idea_lines}",
                }
            ],
        )
        if response.stop_reason == "max_tokens":
            print(f"    Batch {batch_index} hit max_tokens; using partial response")
        parsed = parse_json_response(response.content[0].text, context=f"grouping batch {batch_index}")
        all_clusters.extend(Cluster(**cluster) for cluster in parsed.get("clusters", []))
        time.sleep(0.5)

    print(f"  -> {len(all_clusters)} raw clusters collected")
    if len(batches) > 1:
        return consolidate_clusters(client, all_clusters, model_config=model_config)
    return all_clusters


def consolidate_clusters(
    client: anthropic.Anthropic,
    clusters: list[Cluster],
    *,
    model_config: ModelConfig = DEFAULT_MODELS,
) -> list[Cluster]:
    cluster_lines = "\n".join(
        f"[{index + 1}] THEME: {cluster.theme} | SUMMARY: {cluster.summary}"
        for index, cluster in enumerate(clusters)
    )
    response = client.messages.create(
        model=model_config.grouping_model,
        max_tokens=model_config.grouping_max_tokens,
        system=GROUPING_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Consolidate these {len(clusters)} research clusters by merging "
                    f"near-duplicates and keeping distinct themes. Aim for 25-40 final "
                    f"clusters.\n\n{cluster_lines}\n\nReturn the same JSON schema."
                ),
            }
        ],
    )
    if response.stop_reason == "max_tokens":
        print("    Consolidation hit max_tokens; using partial response")
    parsed = parse_json_response(response.content[0].text, context="cluster consolidation")
    result = [Cluster(**cluster) for cluster in parsed.get("clusters", [])]
    print(f"  -> {len(result)} consolidated clusters")
    return result


def judge_clusters(
    client: anthropic.Anthropic,
    clusters: list[Cluster],
    *,
    batch_size: int = 10,
    model_config: ModelConfig = DEFAULT_MODELS,
) -> list[ScoredCluster]:
    print(f"\n[Judging] Scoring {len(clusters)} clusters...")
    all_scores: list[ScoredCluster] = []
    batches = [clusters[index : index + batch_size] for index in range(0, len(clusters), batch_size)]

    for batch_index, batch in enumerate(batches, start=1):
        print(f"  Batch {batch_index}/{len(batches)} ({len(batch)} clusters)...")
        cluster_text = "\n\n".join(
            f"CLUSTER {index + 1}:\n"
            f"  Theme: {cluster.theme}\n"
            f"  Summary: {cluster.summary}\n"
            f"  Representative idea: {cluster.representative_idea}\n"
            f"  TVB-relevant: {cluster.tvb_relevant}\n"
            f"  Idea count: {cluster.idea_count}"
            for index, cluster in enumerate(batch)
        )
        response = client.messages.create(
            model=model_config.judge_model,
            max_tokens=model_config.judging_max_tokens,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": f"Score these clusters:\n\n{cluster_text}"}],
        )
        if response.stop_reason == "max_tokens":
            print(f"    Judge batch {batch_index} hit max_tokens; using partial response")
        parsed = parse_json_response(response.content[0].text, context=f"judge batch {batch_index}")

        for score_data, cluster in zip(parsed.get("scores", []), batch, strict=False):
            score_data = dict(score_data)
            score_data["total"] = (
                score_data["novelty"]
                + score_data["tvb_feasibility"]
                + score_data["clinical_impact"]
            )
            all_scores.append(
                ScoredCluster(
                    **cluster.model_dump(),
                    **score_data,
                )
            )
        time.sleep(1.0)

    return all_scores


def save_json(ranked: list[ScoredCluster], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump([idea.model_dump() for idea in ranked], handle, indent=2, ensure_ascii=False)
    print(f"  Saved JSON -> {path}")


def save_markdown(ranked: list[ScoredCluster], path: Path, gaps_data: list[dict[str, Any]]) -> None:
    pmid_url = {str(paper["pmid"]): paper.get("url", "") for paper in gaps_data}
    lines = [
        "# Top 20 Research Ideas: Epilepsy x Computational Modeling",
        "",
        "Ranked by combined score (Novelty + TVB Feasibility + Clinical Impact, max 30).",
        "",
        "---",
        "",
    ]

    for rank, idea in enumerate(ranked[:20], start=1):
        lines += [
            f"## #{rank} - {idea.theme}",
            (
                f"**Score: {idea.total}/30** | Novelty: {idea.novelty}/10 | "
                f"TVB Feasibility: {idea.tvb_feasibility}/10 | "
                f"Clinical Impact: {idea.clinical_impact}/10"
            ),
            "",
            f"**Summary:** {idea.summary}",
            "",
            f"**Representative idea:** _{idea.representative_idea}_",
            "",
            f"**Key opportunity:** {idea.key_opportunity}",
            "",
            f"**Rationale:** {idea.rationale}",
            "",
        ]
        if idea.source_pmids:
            links = []
            for pmid in idea.source_pmids[:5]:
                url = pmid_url.get(str(pmid)) or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                links.append(f"[PMID:{pmid}]({url})")
            lines += [f"**Source papers:** {' | '.join(links)}", ""]
        addressability = "TVB-addressable" if idea.tvb_relevant else "Needs extension"
        lines += [f"_{addressability} | {idea.idea_count or '?'} source ideas_", "", "---", ""]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"  Saved Markdown -> {path}")


def run_ranking(
    *,
    input_json: Path = DEFAULT_PATHS.gaps_json,
    json_out: Path = DEFAULT_PATHS.ranked_ideas_json,
    md_out: Path = DEFAULT_PATHS.ranked_ideas_md,
    api_key: str | None = None,
    model_config: ModelConfig = DEFAULT_MODELS,
) -> list[ScoredCluster]:
    with input_json.open(encoding="utf-8") as handle:
        gaps_data = json.load(handle)

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    ideas = collect_ideas(gaps_data)
    question_count = sum(idea.type == "question" for idea in ideas)
    experiment_count = sum(idea.type == "experiment" for idea in ideas)
    print(f"\n[Collected] {len(ideas)} raw ideas from {len(gaps_data)} papers")
    print(f"  {question_count} open questions, {experiment_count} future experiments")

    clusters = group_ideas(client, ideas, model_config=model_config)
    scored = judge_clusters(client, clusters, model_config=model_config)
    ranked = sorted(scored, key=lambda idea: idea.total, reverse=True)
    save_json(ranked, json_out)
    save_markdown(ranked, md_out, gaps_data)

    print(f"\n{'=' * 60}")
    print("TOP 20 RANKED IDEAS")
    print(f"{'=' * 60}")
    for rank, idea in enumerate(ranked[:20], start=1):
        print(f"  #{rank:2d} [{idea.total:2d}/30] {idea.theme}")
    print(f"{'=' * 60}")
    return ranked


def main_rank_ideas() -> None:
    parser = argparse.ArgumentParser(description="Rank research ideas from gaps.json.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PATHS.gaps_json)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_PATHS.ranked_ideas_json)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_PATHS.ranked_ideas_md)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    run_ranking(
        input_json=args.input,
        json_out=args.json_out,
        md_out=args.md_out,
        api_key=args.api_key,
    )
