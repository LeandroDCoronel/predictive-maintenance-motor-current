import numpy as np


def calculate_anomaly_rate(anomalies):

    total = len(anomalies)
    anomaly_count = np.sum(anomalies)

    rate = anomaly_count / total

    return anomaly_count, rate