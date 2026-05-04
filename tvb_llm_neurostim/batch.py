"""Batch simulation engine with checkpointing."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from tvb_llm_neurostim.config import SimulationConfig
from tvb_llm_neurostim.tvb_runtime import load_tvb_lab

DEFAULT_SIMULATION = SimulationConfig()


def simulate_one(
    patient: dict[str, Any],
    iext_boost: float,
    site_index: int,
    *,
    config: SimulationConfig = DEFAULT_SIMULATION,
) -> dict[str, Any]:
    """Run one patient simulation. This function is process-pool safe."""

    try:
        lab = load_tvb_lab()
        conn = lab.connectivity.Connectivity.from_file()
        conn.weights = patient["weights"]
        conn.configure()

        epileptor = lab.models.Epileptor()
        x0_arr = np.ones(76) * patient["x0"]
        x0_arr[patient["soz_node"]] = patient["x0_soz"]
        epileptor.x0 = x0_arr

        iext_arr = np.ones(76) * 3.1
        iext_arr[site_index] += iext_boost
        epileptor.Iext = iext_arr

        sim = lab.simulator.Simulator(
            model=epileptor,
            connectivity=conn,
            coupling=lab.coupling.Linear(a=np.array([config.baseline_coupling])),
            integrator=lab.integrators.EulerDeterministic(dt=config.dt),
            monitors=[lab.monitors.TemporalAverage(period=1.0)],
            simulation_length=config.simulation_length,
        ).configure()

        (time, data), = sim.run()
        del time
        x1 = data[:, 0, :, 0]
        reward = -float(np.mean(np.var(x1, axis=0)))

        return {
            "id": patient["id"],
            "soz_node": patient["soz_node"],
            "soz_type": patient["soz_type"],
            "reward": round(reward, 4),
            "status": "success",
        }
    except Exception as exc:  # pragma: no cover - exercised only in TVB runtime failures
        return {
            "id": patient["id"],
            "soz_node": patient["soz_node"],
            "soz_type": patient.get("soz_type", "unknown"),
            "reward": None,
            "status": "failed",
            "error": str(exc),
        }


class BatchSimulator:
    """Run cohort simulations and persist incremental checkpoints."""

    def __init__(self, checkpoint_file: str | Path = "checkpoint.json"):
        self.checkpoint_file = Path(checkpoint_file)

    def _load_checkpoint(self) -> dict[str, dict[str, Any]]:
        if not self.checkpoint_file.exists():
            return {}
        with self.checkpoint_file.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _save_checkpoint(self, results: dict[str, dict[str, Any]]) -> None:
        with self.checkpoint_file.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    def run_cohort_study(
        self,
        cohort: list[dict[str, Any]],
        iext_boost: float,
        site_index: int,
        *,
        max_workers: int = 4,
    ) -> list[dict[str, Any]]:
        checkpoint = self._load_checkpoint()
        results = dict(checkpoint)
        pending = [patient for patient in cohort if patient["id"] not in results]
        print(f"Total: {len(cohort)} | Completed: {len(results)} | Pending: {len(pending)}")

        if not pending:
            print("All patients already simulated (checkpoint found).")
            return list(results.values())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(simulate_one, patient, iext_boost, site_index): patient
                for patient in pending
            }
            for future in as_completed(futures):
                result = future.result()
                results[result["id"]] = result
                self._save_checkpoint(results)
                status = "ok" if result["status"] == "success" else "failed"
                print(
                    f"  {status} {result['id']} | soz={result['soz_type']} "
                    f"| reward={result.get('reward', 'ERR')}"
                )

        return list(results.values())
