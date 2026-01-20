import pandas as pd
import os

DATA_PATH = os.path.join("data", "supermarket_clean.csv")

def engineer_features(df=None, training_features=None):
    """
    Perform feature engineering on supermarket data.
    If df is None, load the dataset from CSV (training mode).
    If training_features is provided, align the final columns to match training.
    """
    if df is None:
        df = pd.read_csv(DATA_PATH)

    # -------------------------
    # DROP LEAKAGE COLUMNS
    # -------------------------
    drop_cols = [
        'Invoice ID',
        'Tax 5%',
        'cogs',
        'gross income',
        'gross margin percentage'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # -------------------------
    # DATE FEATURES
    # -------------------------
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df.drop('Date', axis=1, inplace=True)

    # -------------------------
    # TIME FEATURES
    # -------------------------
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
        df['Hour'] = df['Time'].dt.hour
        df.drop('Time', axis=1, inplace=True)

    # -------------------------
    # AUTOMATIC CATEGORICAL ENCODING
    # -------------------------
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # -------------------------
    # ALIGN WITH TRAINING FEATURES (if provided)
    # -------------------------
    if training_features is not None:
        for col in training_features:
            if col not in df.columns:
                df[col] = 0
        df = df[training_features]

    return df
