from tvb.simulator.lab import *
import numpy as np

def run_simulation(x0=-1.6, coupling_a=0.0152, patient_coupling_noise=0.0):
    """
    patient_coupling_noise: 模拟病人间个体差异，对coupling_a加扰动
    """
    epileptor = models.Epileptor()
    epileptor.x0 = np.array([x0])
    
    actual_coupling = coupling_a + patient_coupling_noise
    
    sim = simulator.Simulator(
        model=epileptor,
        connectivity=connectivity.Connectivity.from_file(),
        coupling=coupling.Linear(a=np.array([actual_coupling])),
        integrator=integrators.EulerDeterministic(dt=0.05),
        monitors=[monitors.TemporalAverage(period=1.0)],
        simulation_length=1000.0
    ).configure()
    
    (time, data), = sim.run()
    x1 = data[:, 0, :, 0]
    seizure_intensity = np.mean(np.var(x1, axis=0))
    return -seizure_intensity

def run_robust(x0, n_patients=5, seed=42):
    """
    在n个虚拟病人上测试x0，返回worst-case reward
    """
    np.random.seed(seed)
    noises = np.random.uniform(-0.003, 0.003, n_patients)
    rewards = [run_simulation(x0=x0, patient_coupling_noise=n) for n in noises]
    return min(rewards), np.mean(rewards), rewards

if __name__ == "__main__":
    worst, mean, all_r = run_robust(-1.6)
    print(f"x0=-1.6 | worst={worst:.4f} | mean={mean:.4f}")
    worst, mean, all_r = run_robust(-2.5)
    print(f"x0=-2.5 | worst={worst:.4f} | mean={mean:.4f}")
