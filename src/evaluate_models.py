import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from feature_engineering import engineer_features
import os

OUTPUT_MODEL_PATH = "outputs/models/"
OUTPUT_REPORT_PATH = "outputs/reports/"

def plot_feature_importance(model_data, model_name):
    model = model_data['model']
    feature_names = model_data['features']
    
    # Check if model supports feature importance
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not support feature importance.")
        return
    
    importance = model.feature_importances_
    
    # Safety check
    if len(feature_names) != len(importance):
        raise ValueError("Mismatch between features and importance length")
    
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title(f'{model_name} Feature Importance')
    plt.tight_layout()
    
    os.makedirs(OUTPUT_REPORT_PATH, exist_ok=True)
    file_path = os.path.join(
        OUTPUT_REPORT_PATH, f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
    )
    plt.savefig(file_path)
    plt.close()
    
    print(f"{model_name} feature importance saved to: {file_path}")

def evaluate_models():
    # Load data (to ensure same feature pipeline)
    df = engineer_features()
    X = df.drop('Sales', axis=1)
    
    # -------------------------
    # Evaluate Random Forest
    # -------------------------
    rf_data = joblib.load(os.path.join(OUTPUT_MODEL_PATH, "rf_model.pkl"))
    plot_feature_importance(rf_data, "Random Forest")
    
    # -------------------------
    # Evaluate XGBoost
    # -------------------------
    xgb_data = joblib.load(os.path.join(OUTPUT_MODEL_PATH, "xgb_model.pkl"))
    plot_feature_importance(xgb_data, "XGBoost")

if __name__ == "__main__":
    evaluate_models()
