import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:

    def __init__(self):
        self.model = IsolationForest(
            contamination=0.02,
            random_state=42
        )

    def fit(self, X):
        X = X.reshape(-1, 1)
        self.model.fit(X)

    def predict(self, X):
        X = X.reshape(-1, 1)
        preds = self.model.predict(X)

        # convertir a 0 normal / 1 anomalía
        anomalies = np.where(preds == -1, 1, 0)

        return anomalies