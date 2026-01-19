import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
from feature_engineering import engineer_features
import os

OUTPUT_MODEL_PATH = "outputs/models/"

def train_models():
    # -------------------------
    # 1. Load and prepare data
    # -------------------------
    df = engineer_features()
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # -------------------------
    # 2. Random Forest
    # -------------------------
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print("Random Forest Performance")
    print("------------------------")
    print("RMSE:", rmse_rf)
    print("R²:", r2_rf)
    
    # Save Random Forest model + features
    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
    joblib.dump({'model': rf, 'features': X.columns.tolist()}, os.path.join(OUTPUT_MODEL_PATH, "rf_model.pkl"))
    
    # -------------------------
    # 3. XGBoost
    # -------------------------
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    
    print("\nXGBoost Performance")
    print("------------------")
    print("RMSE:", rmse_xgb)
    print("R²:", r2_xgb)
    
    # Save XGBoost model + features
    joblib.dump({'model': xgb_model, 'features': X.columns.tolist()}, os.path.join(OUTPUT_MODEL_PATH, "xgb_model.pkl"))
    
    print("\nModels saved successfully to:", OUTPUT_MODEL_PATH)

if __name__ == "__main__":
    train_models()
