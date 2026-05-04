"""TVB simulation entry points used by the optimization loops."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from tvb_llm_neurostim.config import SimulationConfig
from tvb_llm_neurostim.tvb_runtime import load_tvb_lab

DEFAULT_SIMULATION = SimulationConfig()


def run_simulation(
    x0: float = DEFAULT_SIMULATION.baseline_x0,
    coupling_a: float = DEFAULT_SIMULATION.baseline_coupling,
    patient_coupling_noise: float = 0.0,
    *,
    config: SimulationConfig = DEFAULT_SIMULATION,
) -> float:
    """Run one intrinsic-parameter TVB simulation and return reward."""

    lab = load_tvb_lab()
    epileptor = lab.models.Epileptor()
    epileptor.x0 = np.array([x0])
    actual_coupling = coupling_a + patient_coupling_noise

    sim = lab.simulator.Simulator(
        model=epileptor,
        connectivity=lab.connectivity.Connectivity.from_file(),
        coupling=lab.coupling.Linear(a=np.array([actual_coupling])),
        integrator=lab.integrators.EulerDeterministic(dt=config.dt),
        monitors=[lab.monitors.TemporalAverage(period=1.0)],
        simulation_length=config.simulation_length,
    ).configure()

    (time, data), = sim.run()
    del time
    x1 = data[:, 0, :, 0]
    seizure_intensity = np.mean(np.var(x1, axis=0))
    return -float(seizure_intensity)


def run_robust(
    x0: float,
    *,
    coupling_a: float = DEFAULT_SIMULATION.baseline_coupling,
    n_patients: int = DEFAULT_SIMULATION.n_patients,
    seed: int = DEFAULT_SIMULATION.seed,
    config: SimulationConfig = DEFAULT_SIMULATION,
) -> tuple[float, float, list[float]]:
    """Evaluate one parameter set across sampled patient coupling noise."""

    rng = np.random.default_rng(seed)
    noises = rng.uniform(config.coupling_noise_low, config.coupling_noise_high, n_patients)
    rewards = [
        run_simulation(
            x0=x0,
            coupling_a=coupling_a,
            patient_coupling_noise=float(noise),
            config=config,
        )
        for noise in noises
    ]
    return min(rewards), float(np.mean(rewards)), rewards


@lru_cache(maxsize=1)
def get_labels() -> list[str]:
    lab = load_tvb_lab()
    conn = lab.connectivity.Connectivity.from_file()
    return conn.region_labels.tolist()


def run_simulation_clinical(
    iext_boost: float = 0.0,
    site_index: int = 9,
    patient_noise: float = 0.0,
    *,
    config: SimulationConfig = DEFAULT_SIMULATION,
) -> float:
    """Run a clinical-style external-stimulation simulation."""

    lab = load_tvb_lab()
    conn = lab.connectivity.Connectivity.from_file()
    epileptor = lab.models.Epileptor()
    iext = np.ones(76) * 3.1
    iext[site_index] += iext_boost
    epileptor.Iext = iext

    sim = lab.simulator.Simulator(
        model=epileptor,
        connectivity=conn,
        coupling=lab.coupling.Linear(a=np.array([config.baseline_coupling + patient_noise])),
        integrator=lab.integrators.EulerDeterministic(dt=config.dt),
        monitors=[lab.monitors.TemporalAverage(period=1.0)],
        simulation_length=config.simulation_length,
    ).configure()

    (time, data), = sim.run()
    del time
    x1 = data[:, 0, :, 0]
    return -float(np.mean(np.var(x1, axis=0)))


def run_robust_clinical(
    iext_boost: float,
    site_index: int,
    *,
    n_patients: int = DEFAULT_SIMULATION.n_patients,
    seed: int = DEFAULT_SIMULATION.seed,
    config: SimulationConfig = DEFAULT_SIMULATION,
) -> tuple[float, float, list[float]]:
    rng = np.random.default_rng(seed)
    noises = rng.uniform(config.coupling_noise_low, config.coupling_noise_high, n_patients)
    rewards = [
        run_simulation_clinical(iext_boost, site_index, float(noise), config=config)
        for noise in noises
    ]
    return min(rewards), float(np.mean(rewards)), rewards


def get_clinical_recommendation(iext_boost: float, site_index: int) -> dict[str, object]:
    labels = get_labels()
    site_name = labels[site_index] if site_index < len(labels) else f"Region {site_index}"
    if iext_boost < 1.0:
        intensity = "low"
    elif iext_boost < 2.5:
        intensity = "moderate"
    else:
        intensity = "high"
    return {
        "iext_boost": round(iext_boost, 2),
        "site_index": site_index,
        "site": site_name,
        "intensity": intensity,
        "description": (
            f"{intensity.capitalize()}-intensity stimulation at {site_name} "
            f"(Iext boost = {iext_boost:.1f})"
        ),
    }


def print_intrinsic_smoke_test() -> None:
    for x0 in (DEFAULT_SIMULATION.baseline_x0, -2.5):
        worst, mean, _ = run_robust(x0)
        print(f"x0={x0:.1f} | worst={worst:.4f} | mean={mean:.4f}")


def print_clinical_smoke_test() -> None:
    worst, mean, _ = run_robust_clinical(0.0, 9)
    print(f"Baseline (no stim): worst={worst:.4f} mean={mean:.4f}")
    worst, mean, _ = run_robust_clinical(2.0, 7)
    print(f"Site 7 boost=2.0:   worst={worst:.4f} mean={mean:.4f}")
