import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

# Cargar los datos
data = pd.read_csv('input/cosechas.csv', sep=';', decimal=',')

# Reemplazar separadores de miles y convertir a float
data['Medio'] = data['Medio'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

# Convertir la columna 'Mes' a datetime
data['Mes'] = pd.to_datetime(data['Mes'], format='%d/%m/%Y')
data.set_index('Mes', inplace=True)

# Preprocesamiento de datos
values = data['Medio'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)
train_size = int(len(scaled_values) * 0.8)
train, test = scaled_values[0:train_size], scaled_values[train_size:]

# Parámetros
forecast_horizon = 120  # 10 años (120 meses)
future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_horizon, freq='MS')
z = 1.96  # Para un intervalo de confianza del 95%

# Escalar la desviación estándar cuadráticamente con el tiempo
scaling_factor = np.sqrt(np.arange(1, forecast_horizon + 1))  # Escalamiento cuadrático para mayor dispersión con el tiempo

# Función para desescalar los valores
def inverse_transform(scaled_data):
    return scaler.inverse_transform(scaled_data.reshape(-1, 1)).ravel()

# Funciones para calcular las métricas de error normalizadas
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

# Resultados de los modelos y las métricas de error
results = {}
errors = {}

# Modelo ARCH
arch_model_instance = arch_model(train, vol='ARCH', p=1)
arch_fit = arch_model_instance.fit(disp='off')
arch_forecast = arch_fit.forecast(horizon=forecast_horizon, method='simulation')

# Predicciones para ARCH
arch_mean_scaled = arch_forecast.simulations.values.mean(axis=1).ravel()[:forecast_horizon]
arch_std_scaled = arch_forecast.simulations.values.std(axis=1).ravel()[:forecast_horizon] * scaling_factor

# Desescalar la media
arch_mean = inverse_transform(arch_mean_scaled)

# Calcular la desviación estándar en la escala original
arch_std = arch_std_scaled * (scaler.data_max_ - scaler.data_min_)

results['ARCH'] = (arch_mean, arch_std)

# Calcular las métricas de error para ARCH
y_test_actual = inverse_transform(test.flatten())
y_test_pred_arch = arch_mean[:len(y_test_actual)]
mae_arch, mse_arch, rmse_arch, mape_arch = calculate_errors(y_test_actual, y_test_pred_arch)
errors['ARCH'] = (mae_arch, mse_arch, rmse_arch, mape_arch)

# (Se repite el proceso similar para los modelos GARCH, GJR-GARCH, LSTM, Random Forest y Perceptron...)

# Crear un DataFrame para almacenar las predicciones
df_predictions = pd.DataFrame({'Fecha': future_dates})

# Añadir predicciones de cada modelo al DataFrame
for model_name, (mean, std) in results.items():
    df_predictions[f'{model_name} Mean'] = mean
    df_predictions[f'{model_name} Lower'] = mean - z * std
    df_predictions[f'{model_name} Upper'] = mean + z * std

# Guardar los resultados en un archivo Excel
with pd.ExcelWriter('output/predicciones_modelos_con_intervalos_y_metricas.xlsx') as writer:
    for model_name in results.keys():
        output_df = pd.DataFrame({
            'Fecha': future_dates,
            'Prediccion': df_predictions[f'{model_name} Mean'],
            'Inferior 95%': df_predictions[f'{model_name} Lower'],
            'Superior 95%': df_predictions[f'{model_name} Upper']
        })
        output_df.to_excel(writer, sheet_name=model_name, index=False)

    # Guardar las métricas de error en una hoja separada
    df_errors = pd.DataFrame(errors, index=['MAE', 'MSE', 'RMSE', 'MAPE']).T
    df_errors.to_excel(writer, sheet_name='Metricas de Error')

print("Predicciones, intervalos de confianza y métricas de error guardados en 'output/predicciones_modelos_con_intervalos_y_metricas.xlsx'")
