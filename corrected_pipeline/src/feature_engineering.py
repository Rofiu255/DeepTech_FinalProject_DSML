import pandas as pd
from typing import Optional

DATA_PATH = "data/supermarket_clean.csv"

def engineer_features(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Feature engineering function that works for:
    - Training (df=None → loads dataset)
    - Inference (df provided → user input)
    """

    # -------------------------
    # LOAD DATA (TRAINING MODE)
    # -------------------------
    if df is None:
        df = pd.read_csv(DATA_PATH)

    df = df.copy()

    # -------------------------
    # DROP LEAKAGE / ID COLUMNS
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
        df.drop(columns="Date", inplace=True)

    # -------------------------
    # TIME FEATURES
    # -------------------------
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df["Hour"] = df["Time"].dt.hour
        df.drop(columns="Time", inplace=True)

    # -------------------------
    # TARGET (TRAINING ONLY)
    # -------------------------
    if "Total" in df.columns and "Sales" not in df.columns:
        df.rename(columns={"Total": "Sales"}, inplace=True)

    # -------------------------
    # CATEGORICAL ENCODING
    # -------------------------
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


if __name__ == "__main__":
    df = engineer_features()
    print(df.head())
    print("Shape:", df.shape)
