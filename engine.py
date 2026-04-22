"""
VirtualCohortEngine: BatchSimulator
Parallel TVB simulation across a virtual patient cohort.
Supports checkpointing to resume interrupted runs.
"""
import numpy as np
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tvb.simulator.lab import *


def _simulate_one(patient, iext_boost, site_index):
    """
    Single simulation task — runs in a subprocess.
    Returns dict with patient id, reward, and metadata.
    """
    try:
        conn = connectivity.Connectivity.from_file()
        conn.weights = patient["weights"]
        conn.configure()

        epileptor = models.Epileptor()
        # SOZ node gets higher epileptogenicity
        x0_arr = np.ones(76) * patient["x0"]
        x0_arr[patient["soz_node"]] = patient["x0_soz"]
        epileptor.x0 = x0_arr

        # External stimulation
        iext_arr = np.ones(76) * 3.1
        iext_arr[site_index] += iext_boost
        epileptor.Iext = iext_arr

        sim = simulator.Simulator(
            model=epileptor,
            connectivity=conn,
            coupling=coupling.Linear(a=np.array([0.0152])),
            integrator=integrators.EulerDeterministic(dt=0.05),
            monitors=[monitors.TemporalAverage(period=1.0)],
            simulation_length=1000.0
        ).configure()

        (time, data), = sim.run()
        x1 = data[:, 0, :, 0]
        reward = -float(np.mean(np.var(x1, axis=0)))

        return {
            "id": patient["id"],
            "soz_node": patient["soz_node"],
            "soz_type": patient["soz_type"],
            "reward": round(reward, 4),
            "status": "success"
        }
    except Exception as e:
        return {
            "id": patient["id"],
            "soz_node": patient["soz_node"],
            "soz_type": patient.get("soz_type", "unknown"),
            "reward": None,
            "status": "failed",
            "error": str(e)
        }


class BatchSimulator:
    def __init__(self, checkpoint_file="checkpoint.json"):
        self.checkpoint_file = checkpoint_file

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, results):
        with open(self.checkpoint_file, "w") as f:
            json.dump(results, f)

    def run_cohort_study(self, cohort, iext_boost, site_index, max_workers=4):
        """
        Run simulation for all patients in cohort.
        Supports checkpointing: already-completed patients are skipped.
        """
        checkpoint = self._load_checkpoint()
        results = dict(checkpoint)

        pending = [p for p in cohort if p["id"] not in results]
        print(f"Total: {len(cohort)} | Completed: {len(results)} | Pending: {len(pending)}")

        if not pending:
            print("All patients already simulated (checkpoint found).")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_simulate_one, p, iext_boost, site_index): p
                    for p in pending
                }
                for future in as_completed(futures):
                    res = future.result()
                    results[res["id"]] = res
                    self._save_checkpoint(results)
                    status = "✓" if res["status"] == "success" else "✗"
                    print(f"  {status} {res['id']} | soz={res['soz_type']} | reward={res.get('reward', 'ERR')}")

        return list(results.values())
