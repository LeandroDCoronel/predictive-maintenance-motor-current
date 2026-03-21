import scipy.io
import pandas as pd
import numpy as np
from src.feature_extractor import extract_features

print("Loading dataset...")

data = scipy.io.loadmat("data/raw/motor_current_dataset/ITSC/dataset/Data_ITSC.mat")
itsc = data["itsc"][0,0]

rows = []

window_size = 100

for condition in itsc.dtype.names:

    print("Processing:", condition)

    repetitions = itsc[condition][0,0]

    for rep in repetitions:

        signal_matrix = rep  # shape (1000,3)

        # dividir en ventanas
        for i in range(0, len(signal_matrix), window_size):

            window = signal_matrix[i:i+window_size]

            if len(window) != window_size:
                continue

            features = extract_features(window)

            row = list(features) + [condition]

            rows.append(row)

df = pd.DataFrame(rows)

# nombres de columnas
feature_names = []
signals = ["Ia", "Ib", "Ic"]
stats = ["mean","std","max","min","rms","skew","kurtosis","energy"]

for s in signals:
    for st in stats:
        feature_names.append(f"{st}_{s}")

feature_names.append("condition")

df.columns = feature_names

print(df.head())

df.to_csv("features_dataset.csv", index=False)

print("Dataset with features ready")
print("Total samples:", len(df))