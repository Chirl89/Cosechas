import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import warnings
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA  # Importar la librería ARIMA

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

# Desescalar datos de entrenamiento para modelos que no requieren normalización
train_original = data['Medio'].values[:train_size]
test_original = data['Medio'].values[train_size:]

# Parámetros
forecast_horizon = 120  # 10 años (120 meses)
future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_horizon, freq='MS')
z = 1.96  # Para un intervalo de confianza del 95%

# Escalar la desviación estándar cuadráticamente con el tiempo
scaling_factor = np.sqrt(np.arange(1, forecast_horizon + 1))


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

# Modelo ARIMA
arima_model = ARIMA(train_original, order=(0, 1, 0))
arima_fit = arima_model.fit()

# Predicciones y bandas de confianza para ARIMA
arima_forecast = arima_fit.get_forecast(steps=forecast_horizon)
arima_pred = arima_forecast.predicted_mean
arima_conf_int = arima_forecast.conf_int()

# Cambiar de pandas a NumPy para evitar errores de indexación
arima_lower = arima_conf_int[:, 0]
arima_upper = arima_conf_int[:, 1]

# Guardar resultados de ARIMA
results['ARIMA'] = (arima_pred, (arima_upper - arima_pred) / z)

# Calcular las métricas de error para ARIMA
y_test_pred_arima = arima_fit.forecast(steps=len(test_original))
mae_arima, mse_arima, rmse_arima, mape_arima = calculate_errors(test_original, y_test_pred_arima)
errors['ARIMA'] = (mae_arima, mse_arima, rmse_arima, mape_arima)

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

# Modelo GARCH(1,1)
garch_model_instance = arch_model(train, vol='Garch', p=1, q=1)
garch_fit = garch_model_instance.fit(disp='off')
garch_forecast = garch_fit.forecast(horizon=forecast_horizon, method='simulation')

# Predicciones para GARCH
garch_mean_scaled = garch_forecast.simulations.values.mean(axis=1).ravel()[:forecast_horizon]
garch_std_scaled = garch_forecast.simulations.values.std(axis=1).ravel()[:forecast_horizon] * scaling_factor

# Desescalar la media
garch_mean = inverse_transform(garch_mean_scaled)

# Calcular la desviación estándar en la escala original
garch_std = garch_std_scaled * (scaler.data_max_ - scaler.data_min_)

results['GARCH'] = (garch_mean, garch_std)

# Calcular las métricas de error para GARCH
y_test_pred_garch = garch_mean[:len(y_test_actual)]
mae_garch, mse_garch, rmse_garch, mape_garch = calculate_errors(y_test_actual, y_test_pred_garch)
errors['GARCH'] = (mae_garch, mse_garch, rmse_garch, mape_garch)

# Modelo GJR-GARCH
gjr_model_instance = arch_model(train, vol='Garch', p=1, o=1, q=1)
gjr_fit = gjr_model_instance.fit(disp='off')
gjr_forecast = gjr_fit.forecast(horizon=forecast_horizon, method='simulation')

# Predicciones para GJR-GARCH
gjr_mean_scaled = gjr_forecast.simulations.values.mean(axis=1).ravel()[:forecast_horizon]
gjr_std_scaled = gjr_forecast.simulations.values.std(axis=1).ravel()[:forecast_horizon] * scaling_factor

# Desescalar la media
gjr_mean = inverse_transform(gjr_mean_scaled)

# Calcular la desviación estándar en la escala original
gjr_std = gjr_std_scaled * (scaler.data_max_ - scaler.data_min_)

results['GJR-GARCH'] = (gjr_mean, gjr_std)

# Calcular las métricas de error para GJR-GARCH
y_test_pred_gjr = gjr_mean[:len(y_test_actual)]
mae_gjr, mse_gjr, rmse_gjr, mape_gjr = calculate_errors(y_test_actual, y_test_pred_gjr)
errors['GJR-GARCH'] = (mae_gjr, mse_gjr, rmse_gjr, mape_gjr)

# Modelo LSTM con Dropout y predicción personalizada para Dropout durante la predicción
inputs = Input(shape=(1, 1))
x = LSTM(50, return_sequences=True)(inputs)
x = Dropout(0.6)(x)
x = LSTM(50, return_sequences=False)(x)
x = Dropout(0.6)(x)
outputs = Dense(1)(x)

model_lstm = Model(inputs=inputs, outputs=outputs)
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

X_train_lstm = np.reshape(train, (train.shape[0], 1, 1))
model_lstm.fit(X_train_lstm, train, epochs=20, batch_size=1, verbose=0)

# Función para predecir con Dropout activado (Monte Carlo Dropout)
def lstm_predict_with_uncertainty(model, X, n_iter=100):
    preds = [model(X, training=True) for _ in range(n_iter)]
    preds = np.array(preds)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    return pred_mean.ravel(), pred_std.ravel()

X_forecast_lstm = np.reshape(train[-1], (1, 1, 1))
lstm_predictions = []
lstm_std_predictions = []

for _ in range(forecast_horizon):
    pred_mean, pred_std = lstm_predict_with_uncertainty(model_lstm, X_forecast_lstm)
    lstm_predictions.append(pred_mean[0])
    lstm_std_predictions.append(pred_std[0])
    X_forecast_lstm = np.reshape(pred_mean, (1, 1, 1))

lstm_pred = inverse_transform(np.array(lstm_predictions))
lstm_std = np.array(lstm_std_predictions) * (scaler.data_max_ - scaler.data_min_) * scaling_factor  # Aumentar la incertidumbre con el tiempo
results['LSTM'] = (lstm_pred, lstm_std)

# Calcular las métricas de error para LSTM
y_test_pred_lstm = lstm_pred[:len(y_test_actual)]
mae_lstm, mse_lstm, rmse_lstm, mape_lstm = calculate_errors(y_test_actual, y_test_pred_lstm)
errors['LSTM'] = (mae_lstm, mse_lstm, rmse_lstm, mape_lstm)

# Modelo Random Forest con retardos
def create_lagged_features(data, n_lags=12):
    df = pd.DataFrame(data, columns=['value'])
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['value'].shift(lag)
    return df.dropna()

lagged_train = create_lagged_features(train)
X_train_rf = lagged_train.drop('value', axis=1)
y_train_rf = lagged_train['value']

rf = RandomForestRegressor(n_estimators=200, max_depth=20)
rf.fit(X_train_rf, y_train_rf)

rf_predictions = []
rf_std_predictions = []  # Para almacenar desviaciones estándar de predicciones de árboles

X_forecast_rf = np.array(train[-12:]).reshape(1, -1)

for _ in range(forecast_horizon):
    predicted_rf = rf.predict(X_forecast_rf)
    rf_predictions.append(predicted_rf[0])

    # Obtener predicciones de cada árbol para calcular la desviación estándar
    individual_tree_preds = np.array([tree.predict(X_forecast_rf) for tree in rf.estimators_])
    rf_std_predictions.append(np.std(individual_tree_preds))

    X_forecast_rf = np.roll(X_forecast_rf, -1)
    X_forecast_rf[0, -1] = predicted_rf

rf_pred = inverse_transform(np.array(rf_predictions))
rf_std = np.array(rf_std_predictions) * (scaler.data_max_ - scaler.data_min_) * scaling_factor  # Aumentar la incertidumbre con el tiempo
results['Random Forest'] = (rf_pred, rf_std)

# Calcular las métricas de error para Random Forest
y_test_pred_rf = rf_pred[:len(y_test_actual)]
mae_rf, mse_rf, rmse_rf, mape_rf = calculate_errors(y_test_actual, y_test_pred_rf)
errors['Random Forest'] = (mae_rf, mse_rf, rmse_rf, mape_rf)

# Modelo Perceptron (MLP)
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
mlp.fit(np.arange(len(train)).reshape(-1, 1), train.ravel())
mlp_predictions = mlp.predict(np.arange(len(train), len(train) + forecast_horizon).reshape(-1, 1))
mlp_pred = inverse_transform(mlp_predictions)
mlp_std = np.std(mlp_pred) * scaling_factor  # Aumentar la incertidumbre con el tiempo
results['Perceptron'] = (mlp_pred, mlp_std)

# Calcular las métricas de error para Perceptron
y_test_pred_mlp = mlp_pred[:len(y_test_actual)]
mae_mlp, mse_mlp, rmse_mlp, mape_mlp = calculate_errors(y_test_actual, y_test_pred_mlp)
errors['Perceptron'] = (mae_mlp, mse_mlp, rmse_mlp, mape_mlp)

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

print(
    "Predicciones, intervalos de confianza y métricas de error guardados en 'output/predicciones_modelos_con_intervalos_y_metricas.xlsx'")
