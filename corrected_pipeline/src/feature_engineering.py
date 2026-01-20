import pandas as pd

DATA_PATH = "data/supermarket_clean.csv"

def engineer_features(df=None):
    """
    Perform feature engineering on supermarket data.
    If df is None, load data from CSV (training mode).
    Otherwise, process the passed DataFrame (inference mode).
    """
    if df is None:
        df = pd.read_csv(DATA_PATH)

    # -------------------------
    # DROP ID / LEAKAGE COLUMNS
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

    return df

# -------------------------
# Test run
# -------------------------
if __name__ == "__main__":
    df = engineer_features()
    print(df.head())
    print(df.dtypes)
