from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def calculate_errors(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Normalizar MAE, MSE y RMSE dividiendo por la media de los valores reales
    mean_actual = np.mean(y_true)
    mae_normalized = mae / mean_actual
    mse_normalized = mse / (mean_actual ** 2)
    rmse_normalized = rmse / mean_actual

    return mae_normalized, mse_normalized, rmse_normalized, mape
