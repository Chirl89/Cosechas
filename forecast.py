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
    for i in range(time_step, len(set_entrenamiento_escalado)):
        X.append(set_entrenamiento_escalado[i - time_step:i, 0])
        Y.append(set_entrenamiento_escalado[i, 0])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Definición del modelo LSTM
    modelo = Sequential()
    modelo.add(Input(shape=(X.shape[1], 1)))
    modelo.add(LSTM(units=5, return_sequences=False))
    modelo.add(Dense(units=1))
    modelo.compile(optimizer='adam', loss='mse')

    # Entrenamiento del modelo
    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    modelo.fit(X, Y, epochs=5, batch_size=16, verbose=0, callbacks=[early_stopping])

    # Evaluación del modelo en el conjunto de entrenamiento para obtener errores
    predicciones_entrenamiento = modelo.predict(X)
    errores = Y - predicciones_entrenamiento.flatten()

    # Calcular la desviación estándar de los errores
    desviacion_estandar = np.std(errores)

    # Preparación para la predicción
    set_entrenamiento_escalado = sc.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step, 1))

    # Limpiar sesión para evitar acumulación de gráficos
    clear_session()

    # Predicción iterativa
    predicciones = []
    for _ in range(horizon):
        prediccion = modelo.predict(ultimo_bloque)
        predicciones.append(prediccion[0, 0])

        # Actualizar ultimo_bloque para incluir la nueva predicción
        prediccion_reshaped = np.reshape(prediccion, (1, 1, 1))
        nuevo_bloque = np.append(ultimo_bloque[:, 1:, :], prediccion_reshaped, axis=1)
        ultimo_bloque = nuevo_bloque

    # Convertir predicciones y errores a arrays para el cálculo de bandas
    predicciones = np.array(predicciones)

    # Calcular las bandas superior e inferior en la escala normalizada
    banda_superior = predicciones + 2 * desviacion_estandar
    banda_inferior = predicciones - 2 * desviacion_estandar

    # Desescalar las predicciones y las bandas
    predicciones_desescaladas = sc.inverse_transform(predicciones.reshape(-1, 1)).flatten()
    banda_superior_desescalada = sc.inverse_transform(banda_superior.reshape(-1, 1)).flatten()
    banda_inferior_desescalada = sc.inverse_transform(banda_inferior.reshape(-1, 1)).flatten()

    # Liberar memoria
    del set_entrenamiento_escalado, ultimo_bloque
    gc.collect()

    return predicciones_desescaladas, banda_superior_desescalada, banda_inferior_desescalada


def nn_train_with_bands(vol, horizon, time_step=30):
    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    sc = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    # Crear las secuencias de entrada (X) y las etiquetas (Y) para entrenamiento
    X = []
    Y = []
    for i in range(time_step, len(set_entrenamiento_escalado)):
        X.append(set_entrenamiento_escalado[i - time_step:i, 0])
        Y.append(set_entrenamiento_escalado[i, 0])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1]))

    # Definición del modelo Perceptrón
    modelo = Sequential()
    modelo.add(Input(shape=(X.shape[1],)))
    modelo.add(Dense(units=10, activation='relu'))
    modelo.add(Dense(units=1))
    modelo.compile(optimizer='adam', loss='mse')

    # Entrenamiento del modelo
    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    modelo.fit(X, Y, epochs=10, batch_size=16, verbose=0, callbacks=[early_stopping])

    # Evaluación del modelo en el conjunto de entrenamiento para obtener errores
    predicciones_entrenamiento = modelo.predict(X)
    errores = Y - predicciones_entrenamiento.flatten()

    # Calcular la desviación estándar de los errores
    desviacion_estandar = np.std(errores)

    # Preparación para la predicción
    set_entrenamiento_escalado = sc.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]
    ultimo_bloque = np.reshape(ultimo_bloque, (1, time_step))

    # Limpiar sesión para evitar acumulación de gráficos
    clear_session()

    # Predicción iterativa
    predicciones = []
    for _ in range(horizon):
        prediccion = modelo.predict(ultimo_bloque)
        predicciones.append(prediccion[0, 0])

        # Actualizar ultimo_bloque para incluir la nueva predicción
        ultimo_bloque = np.append(ultimo_bloque[:, 1:], prediccion, axis=1)

    # Convertir predicciones y errores a arrays para el cálculo de bandas
    predicciones = np.array(predicciones)

    # Calcular las bandas superior e inferior en la escala normalizada
    banda_superior = predicciones + 2 * desviacion_estandar
    banda_inferior = predicciones - 2 * desviacion_estandar

    # Desescalar las predicciones y las bandas
    predicciones_desescaladas = sc.inverse_transform(predicciones.reshape(-1, 1)).flatten()
    banda_superior_desescalada = sc.inverse_transform(banda_superior.reshape(-1, 1)).flatten()
    banda_inferior_desescalada = sc.inverse_transform(banda_inferior.reshape(-1, 1)).flatten()

    # Liberar memoria
    del set_entrenamiento_escalado, ultimo_bloque
    gc.collect()

    return predicciones_desescaladas, banda_superior_desescalada, banda_inferior_desescalada


def rf_train_with_bands(vol, horizon, time_step=30):
    # Preprocesamiento de los datos
    set_entrenamiento = vol.to_frame()
    sc = MinMaxScaler(feature_range=(0, 1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    # Crear las secuencias de entrada (X) y las etiquetas (Y) para entrenamiento
    X = []
    Y = []
    for i in range(time_step, len(set_entrenamiento_escalado)):
        X.append(set_entrenamiento_escalado[i - time_step:i, 0])
        Y.append(set_entrenamiento_escalado[i, 0])

    X, Y = np.array(X), np.array(Y)

    # Entrenamiento del modelo Random Forest
    modelo = RandomForestRegressor(n_estimators=500, max_depth=30, min_samples_split=10, min_samples_leaf=4)
    modelo.fit(X, Y)

    # Evaluación del modelo en el conjunto de entrenamiento para obtener errores
    predicciones_entrenamiento = modelo.predict(X)
    errores = Y - predicciones_entrenamiento

    # Calcular la desviación estándar de los errores
    desviacion_estandar = np.std(errores)

    # Preparación para la predicción
    set_entrenamiento_escalado = sc.transform(set_entrenamiento)
    ultimo_bloque = set_entrenamiento_escalado[-time_step:]

    # Predicción iterativa
    predicciones = []
    for _ in range(horizon):
        prediccion = modelo.predict(ultimo_bloque.reshape(1, -1))
        predicciones.append(prediccion[0])

        # Actualizar ultimo_bloque para incluir la nueva predicción
        ultimo_bloque = np.append(ultimo_bloque[1:], prediccion)

    # Convertir predicciones y errores a arrays para el cálculo de bandas
    predicciones = np.array(predicciones)

    # Calcular las bandas superior e inferior en la escala normalizada
    banda_superior = predicciones + 2 * desviacion_estandar
    banda_inferior = predicciones - 2 * desviacion_estandar

    # Desescalar las predicciones y las bandas
    predicciones_desescaladas = sc.inverse_transform(predicciones.reshape(-1, 1)).flatten()
    banda_superior_desescalada = sc.inverse_transform(banda_superior.reshape(-1, 1)).flatten()
    banda_inferior_desescalada = sc.inverse_transform(banda_inferior.reshape(-1, 1)).flatten()

    # Liberar memoria
    del set_entrenamiento_escalado, ultimo_bloque
    gc.collect()

    return predicciones_desescaladas, banda_superior_desescalada, banda_inferior_desescalada