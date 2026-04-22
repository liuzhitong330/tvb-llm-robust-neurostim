from tvb.simulator.lab import *
import numpy as np

REGION_LABELS = None

def get_labels():
    global REGION_LABELS
    if REGION_LABELS is None:
        conn = connectivity.Connectivity.from_file()
        REGION_LABELS = conn.region_labels.tolist()
    return REGION_LABELS

def run_simulation_clinical(iext_boost=0.0, site_index=9, patient_noise=0.0):
    conn = connectivity.Connectivity.from_file()
    epileptor = models.Epileptor()
    iext = np.ones(76) * 3.1
    iext[site_index] += iext_boost
    epileptor.Iext = iext
    sim = simulator.Simulator(
        model=epileptor,
        connectivity=conn,
        coupling=coupling.Linear(a=np.array([0.0152 + patient_noise])),
        integrator=integrators.EulerDeterministic(dt=0.05),
        monitors=[monitors.TemporalAverage(period=1.0)],
        simulation_length=1000.0
    ).configure()
    (time, data), = sim.run()
    x1 = data[:, 0, :, 0]
    return -float(np.mean(np.var(x1, axis=0)))

def run_robust_clinical(iext_boost, site_index, n_patients=5, seed=42):
    np.random.seed(seed)
    noises = np.random.uniform(-0.003, 0.003, n_patients)
    rewards = [run_simulation_clinical(iext_boost, site_index, n) for n in noises]
    return min(rewards), np.mean(rewards), rewards

def get_clinical_recommendation(iext_boost, site_index):
    labels = get_labels()
    site_name = labels[site_index] if site_index < len(labels) else f"Region {site_index}"
    # 将Iext boost映射到临床可读的描述
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
        "description": f"{intensity.capitalize()}-intensity stimulation at {site_name} (Iext boost = {iext_boost:.1f})"
    }

if __name__ == "__main__":
    r, m, _ = run_robust_clinical(0.0, 9)
    print(f"Baseline (no stim): worst={r:.4f} mean={m:.4f}")
    r, m, _ = run_robust_clinical(2.0, 7)
    print(f"Site 7 boost=2.0:   worst={r:.4f} mean={m:.4f}")
