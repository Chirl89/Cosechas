import pandas as pd
import numpy as np

# Cargar datos desde un archivo CSV
input_path = 'input/inputBanorte.csv'
df = pd.read_csv(input_path, delimiter=';', index_col='Mes', parse_dates=['Mes'], dayfirst=True)

# Procesar los valores de la columna 'Medio'
df['Medio'] = df['Medio'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()

# Calcular los retornos porcentuales
df['Retorno'] = df['Medio'].pct_change().dropna()

# Estadísticas básicas
media_retorno = df['Retorno'].mean()
std_retorno = df['Retorno'].std()

# Configuraciones para la simulación
num_simulaciones = 1000
horizonte = 500  # 500 meses hacia atrás

# Inicialización de las simulaciones
simulaciones = np.zeros((horizonte, num_simulaciones))

# Valor inicial (primer valor de la serie original)
primer_valor = df['Medio'].iloc[0]

# Simulando hacia atrás
for t in range(horizonte):
    if t == 0:
        simulaciones[t] = primer_valor / (1 + np.random.normal(media_retorno, std_retorno, num_simulaciones))
    else:
        simulaciones[t] = simulaciones[t-1] / (1 + np.random.normal(media_retorno, std_retorno, num_simulaciones))

# Calcular la media de las simulaciones para cada mes
media_simulaciones = simulaciones.mean(axis=1)

# Calcular la desviación absoluta de cada simulación respecto a la media
desviacion_absoluta = np.abs(simulaciones - media_simulaciones[:, np.newaxis])

# Identificar el 20% de simulaciones más alejadas de la media
percentil_80 = np.percentile(desviacion_absoluta, 80, axis=1)

# Filtrar las simulaciones, conservando las que están dentro del 80% más cercano a la media
simulaciones_filtradas = simulaciones[:, np.all(desviacion_absoluta <= percentil_80[:, np.newaxis], axis=0)]

# Seleccionar una simulación al azar de entre las filtradas
simulacion_elegida = simulaciones_filtradas[:, np.random.choice(simulaciones_filtradas.shape[1])]

# Crear un DataFrame con la simulación elegida y las fechas simuladas
fechas_iniciales = pd.date_range(end=df.index[0], periods=horizonte+1, freq='MS')[::-1][1:]
simulacion_df = pd.DataFrame({'Mes': fechas_iniciales, 'Medio': simulacion_elegida})

# Eliminar la columna 'Retorno' del DataFrame original y preparar para concatenación
df = df.drop(columns=['Retorno'])
df = df.reset_index()

# Concatenar las simulaciones con los datos originales
df_completo = pd.concat([simulacion_df, df])

# Ordenar por 'Mes' de menor a mayor
df_completo = df_completo.sort_values(by='Mes')

# Asegurarse de que los números se manejen correctamente y guardarlos en CSV con el separador ';'
df_completo.to_csv('output/dataframe_completo.csv', sep=';', index=False, float_format='%.2f', decimal=',', encoding='utf-8-sig')

# Mostrar el DataFrame combinado
print(df_completo)
