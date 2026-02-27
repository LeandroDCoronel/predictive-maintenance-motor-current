# tests/test_realistic.py
from src.soft_sensor import run_analysis, simulate_current_data

current, _ = simulate_current_data(n_samples=200, base_current=10, noise_std=0.8, anomaly_prob=0.8)
result = run_analysis(current, voltage=380, cost_per_hour=5000)
print(result)