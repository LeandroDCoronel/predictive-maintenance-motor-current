import joblib
import pandas as pd

print("Loading model...")

model = joblib.load("models/motor_fault_model.pkl")

# ejemplo de corrientes del motor
Ia = float(input("Ia: "))
Ib = float(input("Ib: "))
Ic = float(input("Ic: "))

X = pd.DataFrame([[Ia, Ib, Ic]], columns=["Ia","Ib","Ic"])

prediction = model.predict(X)[0]
prob = model.predict_proba(X).max()

print("\nResult:")
print("Detected condition:", prediction)
print("Confidence:", round(prob*100,2), "%")