import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import os

# Chargement du modèle et du scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "xgb_fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

app = FastAPI(title="Fraud Detection API", version="1.0")

# Schéma d'entrée — 30 features (Time, V1-V28, Amount)
class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost", "version": "1.0"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Convertit en array numpy
    features = np.array([[
        transaction.Time,
        transaction.V1, transaction.V2, transaction.V3, transaction.V4,
        transaction.V5, transaction.V6, transaction.V7, transaction.V8,
        transaction.V9, transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15, transaction.V16,
        transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27, transaction.V28,
        transaction.Amount
    ]])

    # Applique le scaler uniquement sur Time (colonne 0)
    import pandas as pd
    time_df = pd.DataFrame(features[:, [0]], columns=["Time"])
    features[:, 0] = scaler.transform(time_df).flatten()

    # Prédiction
    proba = model.predict_proba(features)[0][1]
    prediction = int(proba >= 0.5)

    return {
        "fraud_probability": round(float(proba), 4),
        "is_fraud": bool(prediction),
        "risk_level": "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.3 else "LOW"
    }