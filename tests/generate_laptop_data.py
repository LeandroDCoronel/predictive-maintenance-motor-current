# tests/generate_laptop_data.py - Generate real laptop data proxy (CPU % as current proxy)

import psutil
import time
import pandas as pd

print("Collecting real laptop data (CPU % proxy)... Press Ctrl+C to stop early.")

data = []
try:
    for i in range(200):
        cpu = psutil.cpu_percent(interval=0.5)
        data.append(cpu)
        print(f"Sample {i+1}: CPU {cpu:.1f}%")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Stopped by user.")

df = pd.DataFrame({'current_proxy': data})
df.to_csv('data/laptop_cpu_proxy.csv', index=False)
print("Data saved to data/laptop_cpu_proxy.csv")