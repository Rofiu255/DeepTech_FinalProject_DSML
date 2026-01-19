import pandas as pd

PROCESSED_PATH = r"data/processed/supermarket_clean.csv"

def engineer_features(input_path=PROCESSED_PATH):
    df = pd.read_csv(input_path)

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    # Convert Time
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour

    # DROP NON-NUMERIC / NON-PREDICTIVE COLUMNS
    drop_cols = [
        'Invoice ID',          # ID column
        'gross margin percentage',  # constant value
        'Date'                 # already extracted
    ]

    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return df


if __name__ == "__main__":
    print(engineer_features().head())
