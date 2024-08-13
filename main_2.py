import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

input_path = f'input/inputBanorte.csv'
df = pd.read_csv(input_path, delimiter=';', index_col='Mes', parse_dates=['Mes'], dayfirst=True)
df['Medio'] = df['Medio'].str.replace('.', '').str.replace(',', '.').astype(float)
df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()

X = df.drop('Medio', axis=1)
y = df['Medio']

# Entrenar el modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predecir los próximos 120 meses
predictions = []
input_data = X.iloc[-1].values.reshape(1, -1)

for _ in range(120):
    pred = model.predict(input_data)[0]
    predictions.append(pred)

    # Actualizar input_data con la nueva predicción
    input_data = np.roll(input_data, -1)
    input_data[0, -1] = pred

# Crear un DataFrame con las predicciones
forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=120, freq='MS')
forecast_df = pd.DataFrame({'Mes': forecast_dates, 'Pronóstico': predictions})
forecast_df.set_index('Mes', inplace=True)

# Exportar a un archivo Excel
file_path = 'pronostico_rf.xlsx'
forecast_df.to_excel(file_path)
