import scipy.io
import pandas as pd

print("Loading dataset...")

path = "data/raw/motor_current_dataset/ITSC/dataset/Data_ITSC.mat"
data = scipy.io.loadmat(path)

itsc = data["itsc"][0,0]

rows = []

conditions = itsc.dtype.names

for condition in conditions:

    experiments = itsc[condition][0,0]

    for exp in experiments:      # 5 experimentos

        for sample in exp:       # 1000 muestras

            ia = float(sample[0])
            ib = float(sample[1])
            ic = float(sample[2])

            rows.append([ia, ib, ic, condition])

df = pd.DataFrame(rows, columns=["Ia","Ib","Ic","condition"])

print(df.head())

df.to_csv("data/processed/motor_current_dataset.csv", index=False)

print("Dataset prepared")
print("Total samples:", len(df))