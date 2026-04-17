#!/usr/bin/env python3
"""
Rank research ideas from gaps.json using Claude.

Pipeline:
  1. Collect all open_questions + future_experiments from gaps.json
  2. Haiku groups them into thematic clusters (deduplication + synthesis)
  3. Opus scores each cluster on novelty, TVB feasibility, clinical impact (1-10)
  4. Top 20 by total score → ranked_ideas.json + ranked_ideas.md
"""

import argparse
import json
import re
import time
import anthropic

HAIKU_MODEL  = "claude-haiku-4-5"
JUDGE_MODEL  = "claude-opus-4-5"


def parse_json_response(text, context="response"):
    """
    Robustly extract a JSON object from a model response.

    Tries in order:
      1. Direct parse after stripping markdown fences
      2. Regex extraction of the outermost {...} block
      3. Truncation repair: close any unclosed arrays/objects and retry
    Raises ValueError if all strategies fail.
    """
    # Strip markdown fences
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    text = text.strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract outermost { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            candidate = match.group()
        # Strategy 3: repair truncated JSON by closing open structures
        repaired = _repair_truncated_json(candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from {context}:\n{text[:300]}")


def _repair_truncated_json(text):
    """
    Close any unclosed JSON arrays and objects so a truncated response
    can be parsed for whatever clusters/scores completed successfully.
    """
    # Remove the last partial element (incomplete string or object)
    # by truncating at the last complete comma-separated item
    text = text.rstrip().rstrip(",")

    # Count unclosed braces/brackets
    depth_brace   = text.count("{") - text.count("}")
    depth_bracket = text.count("[") - text.count("]")

    # Check if we're inside an unclosed string and cut it off cleanly
    # (heuristic: find last unmatched " )
    last_clean = max(text.rfind("}"), text.rfind("]"))
    if last_clean != -1:
        text = text[:last_clean + 1]
        depth_brace   = text.count("{") - text.count("}")
        depth_bracket = text.count("[") - text.count("]")

    text += "]" * max(depth_bracket, 0)
    text += "}" * max(depth_brace,   0)
    return text

# ── Step 1: collect raw ideas ────────────────────────────────────────────────

def collect_ideas(gaps_data):
    ideas = []
    for paper in gaps_data:
        source = {
            "pmid":  paper["pmid"],
            "title": paper["title"],
            "url":   paper["url"],
            "tvb_relevant": paper["gaps"].get("tvb_relevant", False),
        }
        for q in paper["gaps"].get("open_questions", []):
            ideas.append({"text": q, "type": "question", "source": source})
        for e in paper["gaps"].get("future_experiments", []):
            ideas.append({"text": e, "type": "experiment", "source": source})
    return ideas


# ── Step 2: Haiku groups ideas into themes ───────────────────────────────────

GROUPING_SYSTEM = """You are a neuroscience research synthesizer specializing in computational brain modeling and epilepsy.

Given a list of raw research ideas (questions and experiments), group them into coherent thematic clusters.
Each cluster should represent a distinct, actionable research direction relevant to computational neuroscience or epilepsy treatment.

Return ONLY valid JSON — no markdown fences, no commentary — in this exact schema:
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

def group_ideas(client, ideas, batch_size=150):
    """Send ideas to Haiku in batches and collect clusters."""
    print(f"\n[Grouping] {len(ideas)} raw ideas → thematic clusters (Haiku)...")
    all_clusters = []
    batches = [ideas[i:i+batch_size] for i in range(0, len(ideas), batch_size)]

    for b_idx, batch in enumerate(batches, 1):
        print(f"  Batch {b_idx}/{len(batches)} ({len(batch)} ideas)...")

        idea_lines = "\n".join(
            f"[{i+1}] ({item['type']}, PMID:{item['source']['pmid']}) {item['text']}"
            for i, item in enumerate(batch)
        )
        user_msg = f"Group these {len(batch)} research ideas into thematic clusters:\n\n{idea_lines}"

        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=8192,
            system=GROUPING_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        if resp.stop_reason == "max_tokens":
            print(f"    ⚠️  Batch {b_idx} hit max_tokens — using partial response")

        parsed = parse_json_response(resp.content[0].text, context=f"grouping batch {b_idx}")
        all_clusters.extend(parsed.get("clusters", []))
        time.sleep(0.5)

    print(f"  → {len(all_clusters)} raw clusters collected")

    # If multiple batches, do a consolidation pass
    if len(batches) > 1:
        print("  Consolidating clusters across batches...")
        all_clusters = consolidate_clusters(client, all_clusters)

    return all_clusters


def consolidate_clusters(client, clusters):
    """Merge duplicate/overlapping clusters with a second Haiku call."""
    cluster_lines = "\n".join(
        f"[{i+1}] THEME: {c['theme']} | SUMMARY: {c['summary']}"
        for i, c in enumerate(clusters)
    )
    user_msg = (
        f"Consolidate these {len(clusters)} research clusters by merging near-duplicates "
        f"and keeping distinct themes. Aim for 25-40 final clusters.\n\n{cluster_lines}\n\n"
        "Return the same JSON schema as before."
    )
    resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=8192,
        system=GROUPING_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    if resp.stop_reason == "max_tokens":
        print("    ⚠️  Consolidation hit max_tokens — using partial response")
    parsed = parse_json_response(resp.content[0].text, context="consolidation")
    result = parsed.get("clusters", clusters)
    print(f"  → {len(result)} consolidated clusters")
    return result


# ── Step 3: Opus judges each cluster ─────────────────────────────────────────

JUDGE_SYSTEM = """You are a senior neuroscience grant reviewer with deep expertise in:
- Computational brain modeling (The Virtual Brain platform, connectome-based models)
- Drug-resistant epilepsy treatment (resection surgery, brain stimulation, closed-loop systems)
- Translational neuroscience (bench-to-bedside pipeline)

Score each research idea cluster on three dimensions (1–10 each):

NOVELTY (1-10):
  1-3: Well-established, being actively pursued, incremental
  4-6: Some novelty, gaps remain, moderate competition
  7-9: Clearly underexplored, fresh angle, high originality
  10: Genuinely unexplored territory, paradigm-shifting potential

TVB_FEASIBILITY (1-10): How readily can The Virtual Brain (patient-specific whole-brain network model) address this?
  1-3: Requires fundamentally different tools/data not in TVB
  4-6: Possible with TVB extensions or new data pipelines
  7-9: Directly addressable with current/near-term TVB capabilities
  10: TVB is the ideal or only tool for this

CLINICAL_IMPACT (1-10): Potential benefit to drug-resistant epilepsy patients
  1-3: Basic science, distant clinical translation
  4-6: Moderate path to clinical use within 5-10 years
  7-9: High impact, clear clinical application within 5 years
  10: Could immediately change treatment decisions

Return ONLY valid JSON — no markdown, no commentary:
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

def judge_clusters(client, clusters, batch_size=10):
    """Opus scores each cluster. Returns list of scored dicts."""
    print(f"\n[Judging] Scoring {len(clusters)} clusters (Opus)...")
    all_scores = []
    batches = [clusters[i:i+batch_size] for i in range(0, len(clusters), batch_size)]

    for b_idx, batch in enumerate(batches, 1):
        print(f"  Batch {b_idx}/{len(batches)} ({len(batch)} clusters)...")

        cluster_text = "\n\n".join(
            f"CLUSTER {i+1}:\n"
            f"  Theme: {c['theme']}\n"
            f"  Summary: {c['summary']}\n"
            f"  Representative idea: {c['representative_idea']}\n"
            f"  TVB-relevant: {c.get('tvb_relevant', False)}\n"
            f"  Idea count: {c.get('idea_count', '?')}"
            for i, c in enumerate(batch)
        )
        user_msg = f"Score these {len(batch)} research clusters:\n\n{cluster_text}"

        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=4096,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        if resp.stop_reason == "max_tokens":
            print(f"    ⚠️  Judge batch {b_idx} hit max_tokens — using partial response")

        parsed = parse_json_response(resp.content[0].text, context=f"judge batch {b_idx}")
        scores = parsed.get("scores", [])

        # Merge cluster metadata into scores
        for score, cluster in zip(scores, batch):
            score["summary"]            = cluster.get("summary", "")
            score["representative_idea"] = cluster.get("representative_idea", "")
            score["tvb_relevant"]        = cluster.get("tvb_relevant", False)
            score["source_pmids"]        = cluster.get("source_pmids", [])
            score["idea_count"]          = cluster.get("idea_count", 0)
            score["total"]               = score["novelty"] + score["tvb_feasibility"] + score["clinical_impact"]

        all_scores.extend(scores)
        time.sleep(1.0)

    return all_scores


# ── Step 4: render outputs ────────────────────────────────────────────────────

def save_json(ranked, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON → {path}")


def save_markdown(ranked, path, gaps_data):
    # Build pmid → url lookup
    pmid_url = {p["pmid"]: p["url"] for p in gaps_data}

    lines = [
        "# Top 20 Research Ideas: Epilepsy × Computational Modeling",
        "",
        "Ranked by combined score (Novelty + TVB Feasibility + Clinical Impact, max 30).",
        "Judge model: `claude-opus-4-5` · Grouping model: `claude-haiku-4-5`",
        "",
        "---",
        "",
    ]

    for rank, idea in enumerate(ranked[:20], 1):
        n   = idea["novelty"]
        tvb = idea["tvb_feasibility"]
        ci  = idea["clinical_impact"]
        tot = idea["total"]

        lines += [
            f"## #{rank} — {idea['theme']}",
            f"**Score: {tot}/30** &nbsp;|&nbsp; "
            f"Novelty: {n}/10 &nbsp;·&nbsp; TVB Feasibility: {tvb}/10 &nbsp;·&nbsp; Clinical Impact: {ci}/10",
            "",
            f"**Summary:** {idea['summary']}",
            "",
            f"**Representative idea:** _{idea['representative_idea']}_",
            "",
            f"**Key opportunity:** {idea['key_opportunity']}",
            "",
            f"**Rationale:** {idea['rationale']}",
            "",
        ]

        if idea.get("source_pmids"):
            links = []
            for pmid in idea["source_pmids"][:5]:
                url = pmid_url.get(str(pmid), f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                links.append(f"[PMID:{pmid}]({url})")
            lines.append(f"**Source papers:** {' · '.join(links)}")
            lines.append("")

        tvb_tag = "✅ TVB-addressable" if idea.get("tvb_relevant") else "⚙️ Needs extension"
        lines.append(f"_{tvb_tag} · {idea.get('idea_count', '?')} source ideas_")
        lines += ["", "---", ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved Markdown → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",    default="gaps.json",         help="Input gaps JSON")
    parser.add_argument("--json-out", default="ranked_ideas.json", help="Output ranked JSON")
    parser.add_argument("--md-out",   default="ranked_ideas.md",   help="Output ranked Markdown")
    parser.add_argument("--api-key",  default=None,                help="Anthropic API key")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        gaps_data = json.load(f)

    client = anthropic.Anthropic(api_key=args.api_key) if args.api_key else anthropic.Anthropic()

    # 1. Collect
    ideas = collect_ideas(gaps_data)
    print(f"\n[Collected] {len(ideas)} raw ideas from {len(gaps_data)} papers")
    q = sum(1 for i in ideas if i["type"] == "question")
    e = sum(1 for i in ideas if i["type"] == "experiment")
    print(f"  {q} open questions, {e} future experiments")

    # 2. Group
    clusters = group_ideas(client, ideas)

    # 3. Judge
    scored = judge_clusters(client, clusters)

    # 4. Rank + save top 20
    ranked = sorted(scored, key=lambda x: x["total"], reverse=True)

    save_json(ranked, args.json_out)
    save_markdown(ranked, args.md_out, gaps_data)

    # Console summary
    print(f"\n{'='*60}")
    print(f"TOP 20 RANKED IDEAS")
    print(f"{'='*60}")
    for rank, idea in enumerate(ranked[:20], 1):
        print(f"  #{rank:2d} [{idea['total']:2d}/30] {idea['theme']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
