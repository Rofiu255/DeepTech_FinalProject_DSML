import pandas as pd

DATA_PATH = "data/supermarket_clean.csv"

def engineer_features(df=None, *args, **kwargs):
    """
    Robust feature engineering:
    - engineer_features()          → training
    - engineer_features(input_df)  → Streamlit
    """

    # -------------------------
    # LOAD DATA IF NONE PASSED
    # -------------------------
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
        df.drop(columns="Date", inplace=True)

    # -------------------------
    # TIME FEATURES
    # -------------------------
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df["Hour"] = df["Time"].dt.hour
        df.drop(columns="Time", inplace=True)

    # -------------------------
    # ONE-HOT ENCODING
    # -------------------------
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df
