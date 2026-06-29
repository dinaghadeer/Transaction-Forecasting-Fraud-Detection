from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("fraud_model.pkl")
features = joblib.load("features.pkl")
prep = joblib.load("preprocessing.pkl")

amount_mean = prep["amount_mean"]
amount_std = prep["amount_std"]
amount_95 = prep["amount_95"]

THRESHOLD = 0.30


@app.get("/")
def root():
    return {"status": "Fraud Detection API Running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Feature engineering
    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24

    df["log_amount"] = np.log1p(df["amount"])

    df["is_large_tx"] = (
        df["amount"] > amount_95
    ).astype(int)

    df["amount_zscore"] = (
        (df["amount"] - amount_mean)
        / (amount_std + 1e-9)
    )

    df["hour_sin"] = np.sin(
        2*np.pi*df["hour"]/24
    )

    df["hour_cos"] = np.cos(
        2*np.pi*df["hour"]/24
    )

    df = pd.get_dummies(
        df,
        columns=["type"],
        drop_first=True
    )

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    prob = float(model.predict_proba(df)[0][1])

    if prob >= 0.7:
        risk_level = "HIGH"
    elif prob >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": int(prob >= THRESHOLD),
        "risk_level": risk_level
    }
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)