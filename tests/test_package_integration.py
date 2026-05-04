from __future__ import annotations

import subprocess
from pathlib import Path

from tvb_llm_neurostim.config import LiteratureMiningConfig, SimulationConfig
from tvb_llm_neurostim.json_utils import parse_json_response
from tvb_llm_neurostim.rag import infer_reward_trend, retrieve_knowledge
from tvb_llm_neurostim.visualization import render_results_plot

ROOT = Path(__file__).resolve().parents[1]


def test_pydantic_configs_validate_defaults() -> None:
    literature = LiteratureMiningConfig()
    simulation = SimulationConfig()

    assert len(literature.queries) == 10
    assert simulation.coupling_noise_low < simulation.coupling_noise_high
    assert simulation.n_patients == 5


def test_json_response_parser_handles_fences_and_embedded_json() -> None:
    assert parse_json_response('```json\n{"x0": -2.1}\n```') == {"x0": -2.1}
    assert parse_json_response('Reasoning first. {"iext_boost": 0.6, "site_index": 9}') == {
        "iext_boost": 0.6,
        "site_index": 9,
    }


def test_rag_retrieval_prefers_site_and_trend_keywords() -> None:
    kb = [
        {
            "pmid": "1",
            "title": "Hippocampal stimulation for temporal lobe epilepsy",
            "question": "How can suppression be optimized?",
        },
        {
            "pmid": "2",
            "title": "Motor cortex mapping",
            "question": "What biomarkers matter?",
        },
    ]

    retrieved = retrieve_knowledge(kb, "rHC", 0.6, "improving", k=1)
    assert retrieved[0]["pmid"] == "1"
    assert infer_reward_trend([{"reward": -0.5}, {"reward": -0.49}]) == "improving"
    assert infer_reward_trend([{"reward": -0.5}, {"reward": -0.51}]) == "worsening"


def test_legacy_wrappers_import_without_running_tvb() -> None:
    code = "import simulate, simulate_v2, rl_loop, rl_loop_v2, rag_optimizer, bo_comparison"
    result = subprocess.run(
        ["uv", "run", "python", "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_visualization_renders_from_checked_in_results(tmp_path: Path) -> None:
    output = tmp_path / "results.png"
    render_results_plot(ROOT / "results.json", output)
    assert output.exists()
    assert output.stat().st_size > 10_000
