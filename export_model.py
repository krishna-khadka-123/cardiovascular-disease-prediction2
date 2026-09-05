"""
Train the same Logistic Regression pipeline as models.py and export
everything the browser needs (scaler stats, coefficients, metrics)
as a single JSON blob that gets embedded into the HTML file.

To use with your REAL data: replace CSV_PATH below with
'Cardiovascular_Disease.csv' (the real one used in models.py) and re-run.
This will also apply the age/365 conversion + BP filtering used there.
"""
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

CSV_PATH = "synthetic_cardio.csv"
AGE_IN_DAYS = False  # set True if using the real Cardiovascular_Disease.csv (age stored in days)

df = pd.read_csv(CSV_PATH)

if AGE_IN_DAYS:
    df["age"] = df["age"] / 365

df_sample = df[
    (df["ap_hi"] >= 100) & (df["ap_hi"] <= 190) &
    (df["ap_lo"] >= 50) & (df["ap_lo"] <= 99)
]

features = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "alco", "active"]
target = "cardio"

X = df_sample[features]
Y = df_sample[target]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=42, test_size=0.2, stratify=Y
)

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

model = LogisticRegression(
    solver="lbfgs",
    class_weight="balanced",
    random_state=42,
    max_iter=1000,
)
model.fit(X_train_scale, Y_train)
Y_pred = model.predict(X_test_scale)

cr = classification_report(Y_test, Y_pred, output_dict=True)
cm = confusion_matrix(Y_test, Y_pred)

export = {
    "features": features,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "coef": model.coef_[0].tolist(),
    "intercept": float(model.intercept_[0]),
    "classification_report": cr,
    "confusion_matrix": cm.tolist(),
    "accuracy": float(cr["accuracy"]),
}

with open("model_params.json", "w") as f:
    json.dump(export, f, indent=2)

print("Exported model_params.json")
print("Accuracy:", export["accuracy"])
print("Confusion matrix:", cm.tolist())
