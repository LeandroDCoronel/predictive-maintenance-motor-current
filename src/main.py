# main.py - v1 Soft Sensor: Motor Current Analysis

import numpy as np

def simulate_current_data(n_samples=100):
    '''Simulate motor current: normal operation + sudden anomaly at the end.'''
    normal = np.random.normal(10, 1, n_samples - 1)  # 10A mean, small noise
    anomaly = np.array([15.0])  # jump to 15A
    return np.concatenate([normal, anomaly])

def detect_anomaly(current_data, threshold_pct=15):
    '''Detect if last current value exceeds threshold % over historical mean.'''
    mean_current = np.mean(current_data[:-1])
    last_current = current_data[-1]
    increase_pct = ((last_current - mean_current) / mean_current) * 100 if mean_current != 0 else 0
    
    if increase_pct > threshold_pct:
        return True, increase_pct
    return False, increase_pct

def calculate_economic_loss(increase_pct, downtime_hours=48, cost_per_hour=5000):
    '''Simple economic loss estimation based on anomaly severity.'''
    risk_factor = increase_pct / 100
    loss = downtime_hours * cost_per_hour * risk_factor
    return round(loss, 2)

# Run test
if __name__ == '__main__':
    data = simulate_current_data()
    has_anomaly, pct = detect_anomaly(data)
    loss = calculate_economic_loss(pct)
    
    # Definimos la variable aquí para usarla en el print
    downtime_hours = 48  # Puedes cambiar este valor según la fábrica
    
    print(f"Current data (last 5 values): {data[-5:].round(2)} A")
    print(f"Anomaly detected: {has_anomaly} | Increase: {pct:.1f}%")
    print(f"Projected economic loss: ${loss:,} (over {downtime_hours}h downtime)")