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
horizonte = 12  # 12 meses hacia atrás

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

# Crear un DataFrame con los resultados de las simulaciones
simulaciones_df = pd.DataFrame(simulaciones)

# Seleccionar la simulación con el menor error (más cercana al primer valor)
errores = np.abs(simulaciones_df.iloc[0] - primer_valor)
mejor_simulacion = simulaciones_df.iloc[:, errores.argmin()]

# Renombrar la mejor simulación a 'Medio'
mejor_simulacion.name = 'Medio'

# Generar el índice de fechas hacia atrás y luego invertir el orden
fechas_iniciales = pd.date_range(end=df.index[0], periods=horizonte+1, freq='MS')[::-1][1:]
mejor_simulacion.index = fechas_iniciales

# Convertir el índice de mejor_simulacion a una columna llamada 'Mes'
mejor_simulacion = mejor_simulacion.reset_index()
mejor_simulacion = mejor_simulacion.rename(columns={'index': 'Mes'})

# Eliminar la columna 'Retorno' del DataFrame original y preparar para concatenación
df = df.drop(columns=['Retorno'])
df = df.reset_index()

# Concatenar la mejor simulación con los datos originales
df_completo = pd.concat([mejor_simulacion, df])

# Ordenar por 'Mes' de menor a mayor
df_completo = df_completo.sort_values(by='Mes')

# Asegurarse de que los números se manejen correctamente y guardarlos en CSV con el separador ';'
df_completo.to_csv('output/dataframe_completo.csv', sep=';', index=False, float_format='%.2f', decimal=',', encoding='utf-8-sig')

# Mostrar el DataFrame combinado
print(df_completo)
