import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

OUTPUT_MODEL_PATH = "outputs/models/"
OUTPUT_REPORT_PATH = "outputs/reports/"

def plot_feature_importance(model_data, name):
    model = model_data['model']
    features = model_data['features']

    importance = model.feature_importances_

    df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=df)
    plt.title(f"{name} Feature Importance")
    plt.tight_layout()

    os.makedirs(OUTPUT_REPORT_PATH, exist_ok=True)
    plt.savefig(f"{OUTPUT_REPORT_PATH}/{name.lower()}_importance.png")
    plt.close()

    print(f"{name} feature importance saved.")

def evaluate_models():
    rf = joblib.load(f"{OUTPUT_MODEL_PATH}/rf_model.pkl")
    xgb = joblib.load(f"{OUTPUT_MODEL_PATH}/xgb_model.pkl")

    plot_feature_importance(rf, "RandomForest")
    plot_feature_importance(xgb, "XGBoost")

if __name__ == "__main__":
    evaluate_models()
