"""
VirtualCohortEngine: PatientCohort
Generates a heterogeneous virtual patient cohort by applying
biologically plausible perturbations to the base TVB connectome.
"""
import numpy as np
from tvb.simulator.lab import connectivity


class PatientCohort:
    """
    Generates virtual patients with individual variability in:
    - Structural connectivity (white matter tract strength)
    - Seizure onset zone (SOZ) location
    """

    SOZ_REGIONS = {
        "hippocampal": [9, 47],      # rHC, lHC
        "frontal":     [7, 45],      # rFEF, lFEF
        "temporal":    [33, 71],     # rTCS, lTCS
        "occipital":   [35, 73],     # rV1, lV1
    }

    def __init__(self, connectivity_variance=0.1, seed=42):
        """
        Args:
            connectivity_variance: std of multiplicative noise on SC weights
            seed: random seed for reproducibility
        """
        self.base_conn = connectivity.Connectivity.from_file()
        self.base_weights = self.base_conn.weights.copy()
        self.variance = connectivity_variance
        self.rng = np.random.default_rng(seed)

    def _generate_patient(self, patient_id):
        """Generate one virtual patient with perturbed SC and random SOZ."""
        # Structural connectivity perturbation
        noise = self.rng.normal(1.0, self.variance, size=self.base_weights.shape)
        new_weights = self.base_weights * noise
        new_weights = np.clip(new_weights, 0, None)

        # SOZ assignment: sample from clinical distribution
        soz_type = self.rng.choice(
            list(self.SOZ_REGIONS.keys()),
            p=[0.5, 0.2, 0.2, 0.1]  # hippocampal most common
        )
        soz_node = int(self.rng.choice(self.SOZ_REGIONS[soz_type]))

        # x0 perturbation: SOZ node is more epileptogenic
        x0_base = -1.6
        x0_soz = x0_base + self.rng.uniform(0.1, 0.3)  # SOZ closer to threshold

        return {
            "id": f"PATIENT_{patient_id:03d}",
            "weights": new_weights,
            "soz_node": soz_node,
            "soz_type": soz_type,
            "x0": x0_base,
            "x0_soz": x0_soz,
        }

    def generate(self, n_patients=20):
        """
        Generate a cohort of n_patients virtual patients.
        Returns list of patient dicts.
        """
        print(f"Generating cohort of {n_patients} virtual patients "
              f"(SC variance={self.variance})...")
        cohort = [self._generate_patient(i) for i in range(n_patients)]
        soz_dist = {}
        for p in cohort:
            soz_dist[p["soz_type"]] = soz_dist.get(p["soz_type"], 0) + 1
        print(f"SOZ distribution: {soz_dist}")
        return cohort
