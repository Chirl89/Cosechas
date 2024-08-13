import os
import logging
import tensorflow as tf
from sklearn.model_selection import GridSearchCV

# Suprimir advertencias de TensorFlow
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Activation
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import KFold


# 1. Perceptron

def perceptron_train(vol, model_path, horizon, hidden_layer_sizes=(20,), random_state=42, max_iter=5000,
                     learning_rate_init=0.001, window_size=60):
    # Definir la ruta del scaler
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')

    # Preprocesamiento de los datos
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Definir el modelo base
    mlp = MLPRegressor(random_state=random_state, max_iter=max_iter, verbose=0)

    # Definir el grid de hiperparámetros
    param_grid = {
        'hidden_layer_sizes': [(20,), (50,), (100,), (50, 30)],
        'learning_rate_init': [0.001, 0.01, 0.0001],
        'alpha': [0.0001, 0.001, 0.01]  # Regularización L2
    }

    # Implementar GridSearchCV
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)

    # Entrenar el modelo con validación cruzada
    grid_search.fit(X, y)

    # Mejor modelo encontrado por GridSearchCV
    best_mlp = grid_search.best_estimator_

    # Guardar el mejor modelo y el scaler
    joblib.dump(best_mlp, model_path)
    joblib.dump(scaler, scaler_path)

    # Liberar memoria
    del X, y, volatilities_scaled
    gc.collect()


def perceptron_forecast(vol, model, scaler, horizon, window_size=60):
    # Preprocesamiento de los datos
    volatilities_scaled = scaler.transform(vol.values.reshape(-1, 1))
    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = model.predict(last_window)  # Convertir log-volatilidad a volatilidad

    # Invertir la escala de la predicción
    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    # Asegurarse de que la predicción sea no negativa
    predicted_volatility = max(predicted_volatility, 0)

    # Liberar memoria
    del volatilities_scaled, last_window
    gc.collect()

    return predicted_volatility


# 2. LSTM

def lstm_train_simple(vol, model_path, horizon, time_step=3):
    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    sc = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)
    X = np.array([set_entrenamiento_escalado[i - time_step:i, 0] for i in range(time_step, len(set_entrenamiento_escalado) - horizon + 1)])
    Y = set_entrenamiento_escalado[time_step + horizon - 1: len(set_entrenamiento_escalado), 0]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Crear y entrenar el modelo con hiperparámetros fijos
    modelo = Sequential()
    modelo.add(Input(shape=(X.shape[1], 1)))
    modelo.add(LSTM(units=10))
    modelo.add(Dense(units=1))
    modelo.add(Activation('relu'))
    modelo.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    modelo.fit(X, Y, epochs=10, batch_size=1, verbose=0, callbacks=[early_stopping])

    # Guardar el modelo y el scaler
    modelo.save(model_path)
    scaler_path = model_path.replace('.keras', '_scaler.pkl')
    joblib.dump(sc, scaler_path)

    # Liberar memoria
    del X, Y, set_entrenamiento_escalado
    gc.collect()

# Función de predicción con LSTM
def lstm_forecast(vol, model, scaler, horizon, time_step=3):
    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    set_entrenamiento_escalado = scaler.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))

    # Predicción
    prediccion_dia_horizon = model.predict(ultimo_bloque)
    prediccion_dia_horizon = scaler.inverse_transform(prediccion_dia_horizon)

    return prediccion_dia_horizon.flatten()[0]


# 3. Random Forest

def random_forest_train(vol, model_path, horizon, random_state=42, window_size=60):
    # Definir la ruta del scaler
    scaler_path = model_path.replace('.pkl', '_scaler.pkl')

    # Preprocesamiento de los datos
    scaler = StandardScaler()
    volatilities_scaled = scaler.fit_transform(vol.values.reshape(-1, 1))
    X = np.array([volatilities_scaled[i:i + window_size].flatten() for i in
                  range(len(volatilities_scaled) - window_size - horizon + 1)])
    y = volatilities_scaled[window_size + horizon - 1: len(volatilities_scaled)].flatten()

    # Definir el modelo base
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1, verbose=0)

    # Definir el grid de hiperparámetros
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Implementar GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)

    # Entrenar el modelo con validación cruzada
    grid_search.fit(X, y)

    # Mejor modelo encontrado por GridSearchCV
    best_rf = grid_search.best_estimator_

    # Guardar el mejor modelo y el scaler
    joblib.dump(best_rf, model_path)
    joblib.dump(scaler, scaler_path)

    # Liberar memoria
    del X, y, volatilities_scaled
    gc.collect()


def random_forest_forecast(vol, model, scaler, horizon, window_size=60):
    # Preprocesamiento de los datos
    volatilities_scaled = scaler.transform(vol.values.reshape(-1, 1))
    last_window = volatilities_scaled[-window_size:].flatten().reshape(1, -1)
    predicted_volatility = model.predict(last_window)

    # Invertir la escala de la predicción
    predicted_volatility = scaler.inverse_transform(predicted_volatility.reshape(-1, 1)).flatten()[0]

    # Liberar memoria
    del volatilities_scaled, last_window
    gc.collect()

    return predicted_volatility
