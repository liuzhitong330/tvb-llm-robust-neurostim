from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_json(name: str):
    with (RESULTS / name).open(encoding="utf-8") as handle:
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
    landscape = load_json("clinical_landscape.json")
    assert landscape["n_candidates"] == 760
    assert len(landscape["grid"]) == 760
    assert landscape["grid_best"]["site_name"] == "lPFCDM"
    assert landscape["heuristics"]["right_hippocampus"]["site_name"] == "rHC"
    brain = load_json("brain3d_data.json")
    assert len(brain["regions"]) == 76
    assert len(brain["edges"]) > 0
    assert "baseline" in load_json("waveform_data.json")


class SiteParser(__import__("html.parser", fromlist=["HTMLParser"]).HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.images = []
        self.headings = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for name in ("src", "href", "srcset"):
            if name in attrs:
                self.links.append(attrs[name])
        if tag == "img":
            self.images.append(attrs)
        if tag == "h1":
            self.headings += 1


def test_github_pages_site_contains_new_narrative_and_required_assets() -> None:
    from urllib.parse import urlsplit, unquote

    for page in ["index.html", "notebook.html", "brain3d.html"]:
        parser = SiteParser()
        parser.feed((ROOT / page).read_text(encoding="utf-8"))
        assert len(parser.ids) == len(set(parser.ids)), f"Duplicate IDs in {page}"
        for target in parser.links:
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or "${" in target:
                continue
            dest = ROOT / unquote(parts.path) if parts.path else ROOT / page
            assert dest.exists(), f"Missing resource: {page} → {target}"
            if parts.fragment and dest.suffix == ".html":
                linked = SiteParser()
                linked.feed(dest.read_text(encoding="utf-8"))
                assert parts.fragment in linked.ids, f"Broken anchor: {page} → {target}"
        if page == "index.html":
            assert parser.headings == 1
            assert all(image.get("alt") for image in parser.images)
            assert all(image.get("width") and image.get("height") for image in parser.images)
            assert "article" in parser.ids
            assert "patient-readout" in parser.ids


def test_site_chart_targets_are_present_once() -> None:
    # All existing exploratory visualizations remain available in the notebook.
    html = (ROOT / "notebook.html").read_text(encoding="utf-8")
    for chart_id in [
        "waveform-chart", "traj-chart", "forest-chart", "bo-convergence-chart",
        "gen-traintest", "gen-variability", "gen-stress", "brain-iframe",
    ]:
        assert html.count(f'id="{chart_id}"') == 1


def test_editorial_patient_chart_preserves_pairs_and_subgroups() -> None:
    source = load_json("cohort_results_20.json")
    rows = json.loads((ROOT / "assets/essay/cohort-data.json").read_text())
    assert len(rows) == source["n_patients"]
    for i, row in enumerate(rows):
        assert row["id"] == i + 1
        assert row["baseline"] == source["baseline"]["rewards"][i]
        assert row["stimulated"] == source["optimized"]["rewards"][i]
        assert row["subtype"] == source["soz_types"][i]
        assert row["delta"] == round(row["stimulated"] - row["baseline"], 4)
    assert sum(row["delta"] > 0 for row in rows) == 11
    assert sum(row["delta"] < 0 for row in rows) == 9


def test_editorial_figures_are_valid_svg() -> None:
    import xml.etree.ElementTree as ET

    for path in (ROOT / "assets/essay").glob("*.svg"):
        root = ET.parse(path).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
        assert root.find("{http://www.w3.org/2000/svg}desc") is not None
