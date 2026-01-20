import pandas as pd
from typing import Optional

DATA_PATH = "data/supermarket_clean.csv"

def engineer_features(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    # Load dataset if no DataFrame provided (training)
    if df is None:
        df = pd.read_csv(DATA_PATH)

    df = df.copy()

    # -------------------------
    # DROP LEAKAGE COLUMNS
    # -------------------------
    leakage_cols = [
        "Invoice ID",
        "Tax 5%",
        "cogs",
        "gross income",
        "gross margin percentage"
    ]
    df.drop(columns=leakage_cols, inplace=True, errors="ignore")

    # -------------------------
    # DATE FEATURES
    # -------------------------
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        df.drop(columns=["Date"], inplace=True)

    # -------------------------
    # TIME FEATURES
    # -------------------------
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
        df["Hour"] = df["Time"].dt.hour
        df.drop(columns=["Time"], inplace=True)

    # -------------------------
    # ONE-HOT ENCODING
    # -------------------------
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


if __name__ == "__main__":
    df = engineer_features()
    print(df.head())
    print("\nFinal shape:", df.shape)
