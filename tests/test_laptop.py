# tests/test_laptop.py - Test soft sensor with real laptop data

import sys
import os

# Force Python to find 'src' folder from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.soft_sensor import run_analysis
import pandas as pd

# Load CSV from project root's data folder
df = pd.read_csv('data/laptop_cpu_proxy.csv')  # correct relative path from root
current = df['current_proxy'].values

result = run_analysis(
    current,
    voltage=19.5,
    downtime_hours=1.0,
    cost_per_hour=1.0,
    method='zscore'
)
print(result)