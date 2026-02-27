# tests/test_simple.py - Test with fixed anomaly
from src.soft_sensor import run_analysis

# Fixed simulation (original style)
current = np.random.normal(10, 1, 100)
current[-1] = 15.0  # fixed peak

result = run_analysis(current)
print(result)