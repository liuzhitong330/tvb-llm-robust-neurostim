from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(name: str):
    with (ROOT / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_published_intrinsic_result_matches_claim() -> None:
    history = load_json("results.json")
    baseline = history[0]["worst_case_reward"]
    best = max(row["worst_case_reward"] for row in history)
    improvement = (best - baseline) / abs(baseline)

    assert baseline == -0.5285
    assert best == -0.3182
    assert round(improvement * 100, 1) == 39.8 or round(improvement * 100, 1) == 39.9


def test_checked_in_result_artifacts_have_expected_shapes() -> None:
    assert len(load_json("results.json")) == 8
    assert load_json("cohort_results_20.json")["n_patients"] == 20
    assert len(load_json("bo_comparison.json")["bo_trajectory"]) == 8
    assert len(load_json("rag_results.json")["history"]) == 8
    brain = load_json("brain3d_data.json")
    assert len(brain["regions"]) == 76
    assert len(brain["edges"]) > 0
    assert "baseline" in load_json("waveform_data.json")


def test_github_pages_site_contains_new_narrative_and_required_assets() -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert "By <strong>Cathy Liu</strong>" in html
    assert "Language-model-guided search for robust virtual neurostimulation" in html
    assert "Main result and central caveat" in html
    assert "The Pipeline, Without the Jargon" in html
    assert "Parameter Sensitivity Explorer" in html
    assert "What this result is saying" in html
    assert "hero-brain-canvas" in html
    assert "assets/human_brain.glb" in html
    assert (ROOT / "assets" / "human_brain.glb").exists()
    assert "evolve.py" not in html
    assert "github.com/liuzhitong330/tvb-llm-robust-neurostim" in html

    local_sources = re.findall(r'\s(?:src|href)="([^"#:]+)"', html)
    missing = [
        source
        for source in local_sources
        if source
        and "${" not in source
        and not source.startswith(("mailto", "javascript"))
        and not (ROOT / source).exists()
    ]
    assert missing == []


def test_site_chart_targets_are_present_once() -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    for chart_id in [
        "waveform-chart",
        "patient-chart",
        "traj-chart",
        "forest-chart",
        "bo-convergence-chart",
        "gen-traintest",
        "gen-variability",
        "gen-stress",
    ]:
        assert html.count(f'id="{chart_id}"') == 1
