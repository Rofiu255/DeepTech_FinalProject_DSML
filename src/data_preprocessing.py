import pandas as pd
import os

RAW_PATH = r"data/raw/SuperMarket_Analysis.csv"
PROCESSED_PATH = r"data/processed/supermarket_clean.csv"

def preprocess_data(input_path=RAW_PATH, output_path=PROCESSED_PATH):
    # Load data
    df = pd.read_csv(input_path)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert 'Time' to datetime.time
    df['Time'] = pd.to_datetime(df['Time']).dt.time
    
    # Handle missing values (if any)
    df.fillna(method='ffill', inplace=True)
    
    # Encode categorical variables
    categorical_cols = ['Branch','City','Customer type','Gender','Product line','Payment']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Preprocessed data saved to {output_path}")
    return df

if __name__ == "__main__":
    preprocess_data()
