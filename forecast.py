import os
import logging
import tensorflow as tf
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
from tensorflow.keras.backend import clear_session



def lstm_train_with_bands(vol, horizon, time_step=30):
    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    sc = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    # Crear las secuencias de entrada (X) y las etiquetas (Y) para entrenamiento
    X = []
    Y = []
    for i in range(time_step, len(set_entrenamiento_escalado) - horizon):
        X.append(set_entrenamiento_escalado[i - time_step:i, 0])
        Y.append(set_entrenamiento_escalado[i:i + horizon, 0])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Definición del modelo LSTM
    modelo = Sequential()
    modelo.add(Input(shape=(X.shape[1], 1)))
    modelo.add(LSTM(units=50, return_sequences=False))
    modelo.add(Dense(units=horizon))
    modelo.compile(optimizer='adam', loss='mse')

    # Entrenamiento del modelo
    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    modelo.fit(X, Y, epochs=10, batch_size=16, verbose=0, callbacks=[early_stopping])

    # Predicción sobre los últimos datos conocidos
    set_entrenamiento_escalado = sc.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))
    predicciones = modelo.predict(ultimo_bloque)

    # Calcular los errores sobre el conjunto de entrenamiento
    predicciones_entrenamiento = modelo.predict(X)
    errores = Y - predicciones_entrenamiento
    desviacion_estandar = np.std(errores)

    # Calcular las bandas en la escala normalizada
    banda_superior = predicciones + 2 * desviacion_estandar
    banda_inferior = predicciones - 2 * desviacion_estandar

    # Desescalar las predicciones y las bandas
    predicciones_desescaladas = sc.inverse_transform(predicciones).flatten()
    banda_superior_desescalada = sc.inverse_transform(banda_superior).flatten()
    banda_inferior_desescalada = sc.inverse_transform(banda_inferior).flatten()

    return predicciones_desescaladas, banda_superior_desescalada, banda_inferior_desescalada

def nn_train_with_bands(vol, horizon, time_step=30):
    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    sc = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    # Crear las secuencias de entrada (X) y las etiquetas (Y) para entrenamiento
    X = []
    Y = []
    for i in range(time_step, len(set_entrenamiento_escalado) - horizon):
        X.append(set_entrenamiento_escalado[i - time_step:i, 0])
        Y.append(set_entrenamiento_escalado[i:i + horizon, 0])

    X, Y = np.array(X), np.array(Y)

    # Definición del modelo de Red Neuronal
    modelo = Sequential()
    modelo.add(Input(shape=(X.shape[1],)))
    modelo.add(Dense(units=50, activation='relu'))
    modelo.add(Dense(units=horizon))
    modelo.compile(optimizer='adam', loss='mse')

    # Entrenamiento del modelo
    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    modelo.fit(X, Y, epochs=10, batch_size=16, verbose=0, callbacks=[early_stopping])

    # Predicción sobre los últimos datos conocidos
    set_entrenamiento_escalado = sc.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step))
    predicciones = modelo.predict(ultimo_bloque)

    # Calcular los errores sobre el conjunto de entrenamiento
    predicciones_entrenamiento = modelo.predict(X)
    errores = Y - predicciones_entrenamiento
    desviacion_estandar = np.std(errores)

    # Calcular las bandas en la escala normalizada
    banda_superior = predicciones + 2 * desviacion_estandar
    banda_inferior = predicciones - 2 * desviacion_estandar

    # Desescalar las predicciones y las bandas
    predicciones_desescaladas = sc.inverse_transform(predicciones).flatten()
    banda_superior_desescalada = sc.inverse_transform(banda_superior).flatten()
    banda_inferior_desescalada = sc.inverse_transform(banda_inferior).flatten()

    return predicciones_desescaladas, banda_superior_desescalada, banda_inferior_desescalada


def rf_train_with_bands(vol, horizon, time_step=30):
    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    sc = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    # Crear las secuencias de entrada (X) y las etiquetas (Y) para entrenamiento
    X = []
    Y = []
    for i in range(time_step, len(set_entrenamiento_escalado) - horizon):
        X.append(set_entrenamiento_escalado[i - time_step:i, 0])
        Y.append(set_entrenamiento_escalado[i:i + horizon, 0])

    X, Y = np.array(X), np.array(Y)

    # Entrenamiento del modelo Random Forest
    modelo = RandomForestRegressor(n_estimators=500, max_depth=30, min_samples_split=10, min_samples_leaf=4)
    modelo.fit(X, Y)

    # Predicción sobre los últimos datos conocidos
    set_entrenamiento_escalado = sc.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    predicciones = modelo.predict(ultimo_bloque.reshape(1, -1))

    # Calcular los errores sobre el conjunto de entrenamiento
    predicciones_entrenamiento = modelo.predict(X)
    errores = Y - predicciones_entrenamiento
    desviacion_estandar = np.std(errores)

    # Calcular las bandas en la escala normalizada
    banda_superior = predicciones + 2 * desviacion_estandar
    banda_inferior = predicciones - 2 * desviacion_estandar

    # Desescalar las predicciones y las bandas
    predicciones_desescaladas = sc.inverse_transform(predicciones).flatten()
    banda_superior_desescalada = sc.inverse_transform(banda_superior).flatten()
    banda_inferior_desescalada = sc.inverse_transform(banda_inferior).flatten()

    return predicciones_desescaladas, banda_superior_desescalada, banda_inferior_desescalada