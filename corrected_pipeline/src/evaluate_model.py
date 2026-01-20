import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

OUTPUT_MODEL_PATH = "outputs/models/"
OUTPUT_REPORT_PATH = "outputs/reports/"
os.makedirs(OUTPUT_REPORT_PATH, exist_ok=True)

def plot_feature_importance(model, features, name):
    importance = model.feature_importances_

    df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=df)
    plt.title(f"{name} Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_REPORT_PATH}/{name.lower()}_importance.png")
    plt.close()

    print(f"{name} feature importance saved.")

def evaluate_models():
    rf = joblib.load(f"{OUTPUT_MODEL_PATH}/random_forest_model.pkl")
    xgb = joblib.load(f"{OUTPUT_MODEL_PATH}/xgb_model.pkl")
    features = joblib.load(f"{OUTPUT_MODEL_PATH}/training_features.pkl")

    plot_feature_importance(rf, features, "RandomForest")
    plot_feature_importance(xgb, features, "XGBoost")

if __name__ == "__main__":
    evaluate_models()
