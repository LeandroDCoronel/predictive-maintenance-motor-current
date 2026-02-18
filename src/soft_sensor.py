# soft_sensor.py - Core module for motor current analysis (MCSA)

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomaly(current_data, method='isolation_forest', threshold_pct=15, contamination=0.05):
    """
    Detect anomalies in current data series.

    Args:
        current_data (array-like): 1D array of current values in Amperes
        method (str): 'isolation_forest' (default) or 'zscore'
        threshold_pct (float): % increase threshold for zscore method
        contamination (float): expected anomaly fraction for IsolationForest

    Returns:
        tuple: (has_anomaly: bool, max_pct_increase: float, anomaly_index: int or None)
    """
    current_data = np.asarray(current_data)
    
    if method == 'isolation_forest':
        df = pd.DataFrame({'current': current_data})
        model = IsolationForest(contamination=contamination, random_state=42)
        df['anomaly_score'] = model.fit_predict(df[['current']])
        anomalies = df[df['anomaly_score'] == -1]
        if not anomalies.empty:
            idx = anomalies.index[0]
            pct = ((current_data[idx] - np.mean(current_data)) / np.mean(current_data)) * 100
            return True, pct, idx
        return False, 0.0, None
    
    elif method == 'zscore':
        mean = np.mean(current_data)
        std = np.std(current_data)
        z_scores = (current_data - mean) / std
        max_z = z_scores.max()
        if max_z > 3:
            idx = np.argmax(z_scores)
            pct = (max_z * std / mean) * 100 if mean != 0 else 0.0
            return True, pct, idx
        return False, 0.0, None
    
    else:
        raise ValueError("method must be 'isolation_forest' or 'zscore'")


def calculate_power(current_data, voltage=380.0, power_factor=0.85):
    """Calculate power in kW: P = V × I × cosφ (simplified three-phase)"""
    return voltage * np.asarray(current_data) * power_factor / 1000.0


def calculate_economic_loss(max_pct_increase, downtime_hours=48.0, cost_per_hour=5000.0):
    """Estimate economic loss in USD based on anomaly severity"""
    risk_factor = max_pct_increase / 100.0
    loss = downtime_hours * cost_per_hour * risk_factor
    return round(loss, 2)


def run_analysis(current_data, voltage=380.0, downtime_hours=48.0, cost_per_hour=5000.0, method='isolation_forest'):
    """
    Run full analysis pipeline.

    Args:
        current_data (array-like): current values in Amperes
        voltage (float): operating voltage in Volts
        downtime_hours (float): projected downtime hours
        cost_per_hour (float): cost per downtime hour in USD
        method (str): detection method

    Returns:
        dict: analysis results
    """
    has_anomaly, max_pct, anomaly_idx = detect_anomaly(current_data, method=method)
    power_data = calculate_power(current_data, voltage)
    loss = calculate_economic_loss(max_pct, downtime_hours, cost_per_hour) if has_anomaly else 0.0
    
    return {
        'has_anomaly': has_anomaly,
        'max_pct_increase': max_pct,
        'anomaly_index': anomaly_idx,
        'max_power_kw': power_data.max(),
        'economic_loss_usd': loss,
        'summary': f"Anomaly: {'Detected' if has_anomaly else 'None'} | Loss: ${loss:,.2f}" if has_anomaly else "No anomaly detected"
    }

def save_report(results, filename='data/analysis_report.csv'):
    """
    Save the analysis results to a CSV file.
    
    Args:
        results (dict): Output from run_analysis()
        filename (str): Path to save the CSV (default: data/analysis_report.csv)
    """
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)
    print(f"Report saved to {filename}")