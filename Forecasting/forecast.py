"""
forecast.py
───────────
Recursive multi-step forecast using a trained LightGBM pair.
"""

import pandas as pd
import numpy as np

LAGS = [1, 2, 3, 6, 12, 24]


def recursive_forecast(
    history_df: pd.DataFrame,
    model_amount,
    model_count,
    feature_cols: list,
    steps: int = 48,
) -> list:
    """
    Parameters
    ----------
    history_df   : DataFrame with columns [Datetime, Amount, tx_count]
                   must have at least 24 rows (hourly resolution)
    model_amount : trained LGBMRegressor for Amount
    model_count  : trained LGBMRegressor for tx_count
    feature_cols : list of feature column names (from feature_columns.pkl)
    steps        : how many hours ahead to forecast

    Returns
    -------
    list of dicts: [{"Time": ISO str, "pred_amount": float, "pred_count": float}, ...]
    """
    df = history_df.copy()
    results = []

    for _ in range(steps):
        last_row  = df.iloc[-1]
        next_time = pd.Timestamp(last_row["Datetime"]) + pd.Timedelta(hours=1)

        row = {
            "Datetime":    next_time,
            "Amount":      last_row["Amount"],
            "tx_count":    last_row["tx_count"],
            "avg_ticket":  last_row["Amount"] / max(last_row["tx_count"], 1),
            "hour_of_day": next_time.hour,
            "day_of_week": next_time.dayofweek,
        }

        for l in LAGS:
            idx = max(-l, -len(df))
            row[f"lag_amount_{l}"] = float(df["Amount"].iloc[idx])
            row[f"lag_count_{l}"]  = float(df["tx_count"].iloc[idx])

        X = pd.DataFrame([row])[feature_cols]

        pred_amount = float(max(model_amount.predict(X)[0], 0))
        pred_count  = float(max(round(model_count.predict(X)[0]), 0))

        results.append({
            "Time":        next_time.isoformat(),
            "pred_amount": round(pred_amount, 2),
            "pred_count":  int(pred_count),
        })

        new_row = pd.DataFrame([{
            "Datetime": next_time,
            "Amount":   pred_amount,
            "tx_count": pred_count,
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    return results
