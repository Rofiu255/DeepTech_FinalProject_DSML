import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from feature_engineering import engineer_features

# -------------------------
# PATHS
# -------------------------
OUTPUT_MODEL_PATH = "outputs/models"
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)

# -------------------------
# TRAINING FUNCTION
# -------------------------
def train_models():
    # Load engineered (leak-free) dataset
    df = engineer_features()

    # -------------------------
    # TARGET & FEATURES
    # -------------------------
    X = df.drop("Sales", axis=1)
    y = df["Sales"]

    # 🔐 SAVE FEATURE NAMES (CRITICAL FOR STREAMLIT)
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(OUTPUT_MODEL_PATH, "training_features.pkl"))

    # -------------------------
    # TRAIN / TEST SPLIT
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # RANDOM FOREST
    # =========================
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print("\nRandom Forest Performance")
    print("------------------------")
    print("RMSE:", rmse_rf)
    print("R²:", r2_rf)

    # =========================
    # XGBOOST
    # =========================
    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
    xgb.fit(X_train, y_train)

    y_pred_xgb = xgb.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

    print("\nXGBoost Performance")
    print("------------------")
    print("RMSE:", rmse_xgb)
    print("R²:", r2_xgb)

    # -------------------------
    # SAVE MODELS
    # -------------------------
    joblib.dump(rf, os.path.join(OUTPUT_MODEL_PATH, "random_forest_model.pkl"))
    joblib.dump(xgb, os.path.join(OUTPUT_MODEL_PATH, "xgb_model.pkl"))

    print("\nModels saved to:", OUTPUT_MODEL_PATH)

    return {
        "RandomForest": {"RMSE": rmse_rf, "R2": r2_rf},
        "XGBoost": {"RMSE": rmse_xgb, "R2": r2_xgb}
    }

# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    train_models()
