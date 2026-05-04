"""Virtual patient cohort generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from tvb_llm_neurostim.config import CohortConfig
from tvb_llm_neurostim.tvb_runtime import load_tvb_lab

DEFAULT_COHORT = CohortConfig()


class PatientCohort:
    """Generate heterogeneous virtual patients from a base TVB connectome."""

    def __init__(self, config: CohortConfig = DEFAULT_COHORT):
        self.config = config
        lab = load_tvb_lab()
        self.base_conn = lab.connectivity.Connectivity.from_file()
        self.base_weights = self.base_conn.weights.copy()
        self.rng = np.random.default_rng(config.seed)

    def _generate_patient(self, patient_id: int) -> dict[str, Any]:
        noise = self.rng.normal(
            1.0,
            self.config.connectivity_variance,
            size=self.base_weights.shape,
        )
        weights = np.clip(self.base_weights * noise, 0, None)

        soz_types = list(self.config.soz_regions)
        probabilities = [self.config.soz_probabilities[soz_type] for soz_type in soz_types]
        soz_type = str(self.rng.choice(soz_types, p=probabilities))
        soz_node = int(self.rng.choice(self.config.soz_regions[soz_type]))
        x0_soz = self.config.x0_base + self.rng.uniform(0.1, 0.3)

        return {
            "id": f"PATIENT_{patient_id:03d}",
            "weights": weights,
            "soz_node": soz_node,
            "soz_type": soz_type,
            "x0": self.config.x0_base,
            "x0_soz": x0_soz,
        }

    def generate(self, n_patients: int | None = None) -> list[dict[str, Any]]:
        count = n_patients or self.config.n_patients
        print(
            f"Generating cohort of {count} virtual patients "
            f"(SC variance={self.config.connectivity_variance})..."
        )
        cohort = [self._generate_patient(index) for index in range(count)]
        soz_distribution: dict[str, int] = {}
        for patient in cohort:
            soz_distribution[patient["soz_type"]] = soz_distribution.get(patient["soz_type"], 0) + 1
        print(f"SOZ distribution: {soz_distribution}")
        return cohort
