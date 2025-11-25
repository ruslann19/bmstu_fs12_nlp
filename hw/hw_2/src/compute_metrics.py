from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = root_mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
