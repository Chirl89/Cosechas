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

# Escalar la desviación estándar con el tiempo (factor mucho más moderado)
scaling_factor = np.linspace(1, 1.5, forecast_horizon)  # Factor mucho más moderado para reducir la amplitud


# Función para desescalar los valores
def inverse_transform(scaled_data):
    return scaler.inverse_transform(scaled_data.reshape(-1, 1)).ravel()


# Funciones para calcular las métricas de error
def calculate_errors(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, mse, rmse, mape


# Resultados de los modelos y las métricas de error
results = {}
errors = {}

# Modelo ARCH
arch_model_instance = arch_model(train, vol='ARCH', p=1)
arch_fit = arch_model_instance.fit(disp='off')
arch_forecast = arch_fit.forecast(horizon=forecast_horizon, method='simulation')

# Predicciones y desviación estándar para ARCH
arch_mean = inverse_transform(arch_forecast.simulations.values.mean(axis=1).ravel()[:forecast_horizon])
arch_std = inverse_transform(arch_forecast.simulations.values.std(axis=1).ravel()[:forecast_horizon]) * scaling_factor
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

# Predicciones y desviación estándar para GARCH
garch_mean = inverse_transform(garch_forecast.simulations.values.mean(axis=1).ravel()[:forecast_horizon])
garch_std = inverse_transform(garch_forecast.simulations.values.std(axis=1).ravel()[:forecast_horizon]) * scaling_factor
results['GARCH'] = (garch_mean, garch_std)

# Calcular las métricas de error para GARCH
y_test_pred_garch = garch_mean[:len(y_test_actual)]
mae_garch, mse_garch, rmse_garch, mape_garch = calculate_errors(y_test_actual, y_test_pred_garch)
errors['GARCH'] = (mae_garch, mse_garch, rmse_garch, mape_garch)

# Modelo GJR-GARCH
gjr_model_instance = arch_model(train, vol='Garch', p=1, o=1, q=1)
gjr_fit = gjr_model_instance.fit(disp='off')
gjr_forecast = gjr_fit.forecast(horizon=forecast_horizon, method='simulation')

# Predicciones y desviación estándar para GJR-GARCH
gjr_mean = inverse_transform(gjr_forecast.simulations.values.mean(axis=1).ravel()[:forecast_horizon])
gjr_std = inverse_transform(gjr_forecast.simulations.values.std(axis=1).ravel()[:forecast_horizon]) * scaling_factor
results['GJR-GARCH'] = (gjr_mean, gjr_std)

# Calcular las métricas de error para GJR-GARCH
y_test_pred_gjr = gjr_mean[:len(y_test_actual)]
mae_gjr, mse_gjr, rmse_gjr, mape_gjr = calculate_errors(y_test_actual, y_test_pred_gjr)
errors['GJR-GARCH'] = (mae_gjr, mse_gjr, rmse_gjr, mape_gjr)

# Modelo LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
model_lstm.add(LSTM(50, return_sequences=False))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

X_train_lstm = np.reshape(train, (train.shape[0], 1, 1))
model_lstm.fit(X_train_lstm, train, epochs=20, batch_size=1, verbose=0)

X_forecast_lstm = np.reshape(train[-1], (1, 1, 1))
lstm_predictions = []

for _ in range(forecast_horizon):
    predicted_lstm = model_lstm.predict(X_forecast_lstm)
    lstm_predictions.append(predicted_lstm[0, 0])
    X_forecast_lstm = np.reshape(predicted_lstm, (1, 1, 1))

lstm_pred = inverse_transform(np.array(lstm_predictions))
lstm_std = np.std(lstm_pred) * scaling_factor  # Aumentar la incertidumbre con el tiempo
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

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train_rf, y_train_rf)

rf_predictions = []
X_forecast_rf = np.array(train[-12:]).reshape(1, -1)

for _ in range(forecast_horizon):
    predicted_rf = rf.predict(X_forecast_rf)
    rf_predictions.append(predicted_rf[0])
    X_forecast_rf = np.roll(X_forecast_rf, -1)
    X_forecast_rf[0, -1] = predicted_rf

rf_pred = inverse_transform(np.array(rf_predictions))
rf_std = np.std(rf_pred) * scaling_factor  # Aumentar la incertidumbre con el tiempo
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
