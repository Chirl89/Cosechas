import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import jarque_bera

# Cargar los datos
data = pd.read_csv('input/cosechas.csv', sep=';', decimal=',')

# Reemplazar separadores de miles y convertir a float
data['Medio'] = data['Medio'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

# Convertir la columna 'Mes' a datetime
data['Mes'] = pd.to_datetime(data['Mes'], format='%d/%m/%Y')
data.set_index('Mes', inplace=True)

# Realizar el test de Dickey-Fuller
adf_result = adfuller(data['Medio'])

# Realizar el test de Jarque-Bera
jb_result = jarque_bera(data['Medio'])

# Imprimir los resultados
print("Dickey-Fuller Test:")
print(f"Estadístico ADF: {adf_result[0]}")
print(f"p-valor: {adf_result[1]}")

print("\nTest de Jarque-Bera:")
print(f"Estadístico JB: {jb_result[0]}")
print(f"p-valor: {jb_result[1]}")
