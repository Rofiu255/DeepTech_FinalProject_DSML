import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def print_metrics(model_name, metrics):
    print(f"\n{model_name} Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
