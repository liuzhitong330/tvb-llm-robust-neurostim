"""Typed configuration for the research pipeline.

The project still exposes simple scripts at the repository root, but the core
configuration lives here so runs are reproducible and reviewable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrozenModel(BaseModel):
    """Base class for immutable config models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PathsConfig(FrozenModel):
    """Default artifact paths used by the pipeline and GitHub Pages site."""

    papers_csv: Path = Path("all_papers.csv")
    gaps_json: Path = Path("gaps.json")
    ranked_ideas_json: Path = Path("ranked_ideas.json")
    ranked_ideas_md: Path = Path("ranked_ideas.md")
    results_json: Path = Path("results.json")
    results_png: Path = Path("results.png")
    clinical_results_json: Path = Path("results_v2.json")
    cohort_results_json: Path = Path("cohort_results_20.json")


class ModelConfig(FrozenModel):
    """Model names and token budgets for Anthropic calls."""

    haiku_model: str = "claude-haiku-4-5-20251001"
    grouping_model: str = "claude-haiku-4-5"
    judge_model: str = "claude-opus-4-5"
    optimizer_model: str = "claude-opus-4-5"
    extraction_max_tokens: int = Field(default=1024, ge=256)
    grouping_max_tokens: int = Field(default=8192, ge=1024)
    judging_max_tokens: int = Field(default=4096, ge=1024)
    optimization_max_tokens: int = Field(default=512, ge=128)


class LiteratureMiningConfig(FrozenModel):
    """PubMed and LLM literature-mining settings."""

    email: str = "cathyliu014@gmail.com"
    max_per_query: int = Field(default=100, ge=1, le=500)
    request_pause_seconds: float = Field(default=1.0, ge=0)
    extraction_pause_seconds: float = Field(default=0.5, ge=0)
    queries: tuple[str, ...] = (
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
    )


class SimulationConfig(FrozenModel):
    """TVB simulation defaults for intrinsic-parameter optimization."""

    baseline_x0: float = -1.6
    optimized_x0: float = -2.1
    baseline_coupling: float = 0.0152
    optimized_coupling: float = 0.0165
    coupling_noise_low: float = -0.003
    coupling_noise_high: float = 0.003
    n_patients: int = Field(default=5, ge=1)
    seed: int = 42
    simulation_length: float = Field(default=1000.0, gt=0)
    dt: float = Field(default=0.05, gt=0)

    @field_validator("coupling_noise_high")
    @classmethod
    def validate_noise_range(cls, value: float, info) -> float:
        low = info.data.get("coupling_noise_low")
        if low is not None and value < low:
            raise ValueError("coupling_noise_high must be >= coupling_noise_low")
        return value


class CohortConfig(FrozenModel):
    """Virtual cohort generation defaults."""

    connectivity_variance: float = Field(default=0.1, ge=0)
    n_patients: int = Field(default=20, ge=1)
    seed: int = 42
    x0_base: float = -1.6
    soz_regions: dict[str, tuple[int, ...]] = {
        "hippocampal": (9, 47),
        "frontal": (7, 45),
        "temporal": (33, 71),
        "occipital": (35, 73),
    }
    soz_probabilities: dict[str, float] = {
        "hippocampal": 0.5,
        "frontal": 0.2,
        "temporal": 0.2,
        "occipital": 0.1,
    }


class OptimizationConfig(FrozenModel):
    """Shared optimization-loop settings."""

    n_iterations: int = Field(default=8, ge=1)
    baseline_site_index: int = Field(default=9, ge=0, le=75)
    initial_iext_boost: float = Field(default=1.5, ge=0.0, le=4.0)
    initial_site_index: int = Field(default=9, ge=0, le=75)
    best_clinical_iext_boost: float = Field(default=0.6, ge=0.0, le=4.0)
    best_clinical_site_index: int = Field(default=9, ge=0, le=75)
