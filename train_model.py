import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

print("Loading dataset...")

df = pd.read_csv("data/processed/motor_current_dataset.csv")

X = df[["Ia","Ib","Ic"]]
y = df["condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Evaluating...")

pred = model.predict(X_test)

print(classification_report(y_test, pred))

joblib.dump(model, "models/motor_fault_model.pkl")

print("Model saved to models/motor_fault_model.pkl")