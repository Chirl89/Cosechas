import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from forecast import *

scale_factor = 10000000
forecast_horizon = 120
time_step = 3

input_path = f'input/inputBanorte.csv'
df = pd.read_csv(input_path, delimiter=';', index_col='Mes', parse_dates=['Mes'], dayfirst=True)
df['Medio'] = df['Medio'].str.replace('.', '').str.replace(',', '.').astype(float)
df.index = pd.DatetimeIndex(df.index).to_period('M').to_timestamp()

data = df['Medio'].dropna()
# data_scaled = df['Medio'].dropna() / scale_factor

model_arima = ARIMA(data, order=(0, 1, 0))
fitted_arima = model_arima.fit()
residuales = fitted_arima.resid

forecast_arima = fitted_arima.get_forecast(steps=forecast_horizon)
predicted_mean = forecast_arima.predicted_mean

forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')

###################################################
################# ARCH ############################
###################################################

model_arch = arch_model(residuales / scale_factor, vol='ARCH', p=1)
fitted_arch = model_arch.fit(disp='off')

vol_forecast_arch = fitted_arch.forecast(horizon=forecast_horizon)
conditional_vol_arch = vol_forecast_arch.variance[-1:]

predicted_values_arch = predicted_mean + (conditional_vol_arch.values[-1, :] ** 0.5) * scale_factor
upper_band_arch = predicted_mean + 1.96 * (conditional_vol_arch.values[-1, :] ** 0.5) * scale_factor
lower_band_arch = predicted_mean - 1.96 * (conditional_vol_arch.values[-1, :] ** 0.5) * scale_factor

forecast_df_arch = pd.DataFrame({'Mes': forecast_dates,
                                 'Pronóstico': predicted_values_arch,
                                 'Inferior 95': lower_band_arch,
                                 'Superior 95': upper_band_arch})

forecast_df_arch.set_index('Mes', inplace=True)

###################################################
################# GARCH ###########################
###################################################

model_garch = arch_model(residuales / scale_factor, vol='GARCH', p=1, q=1)
fitted_garch = model_garch.fit(disp='off')

vol_forecast_garch = fitted_garch.forecast(horizon=forecast_horizon)
conditional_vol_garch = vol_forecast_garch.variance[-1:]

predicted_values_garch = predicted_mean + (conditional_vol_arch.values[-1, :] ** 0.5) * scale_factor
upper_band_garch = predicted_mean + 1.96 * (conditional_vol_arch.values[-1, :] ** 0.5) * scale_factor
lower_band_garch = predicted_mean - 1.96 * (conditional_vol_arch.values[-1, :] ** 0.5) * scale_factor

forecast_df_garch = pd.DataFrame({'Mes': forecast_dates,
                                  'Pronóstico': predicted_values_garch,
                                  'Inferior 95': lower_band_garch,
                                  'Superior 95': upper_band_garch})

forecast_df_garch.set_index('Mes', inplace=True)

###################################################
################# GJR-GARCH #######################
###################################################

model_gjr_garch = arch_model(residuales / scale_factor, vol='GARCH', p=1, o=1, q=1)
fitted_gjr_garch = model_gjr_garch.fit(disp='off')

vol_forecast_gjr_garch = fitted_gjr_garch.forecast(horizon=forecast_horizon)
conditional_vol_gjr_garch = vol_forecast_gjr_garch.variance[-1:]

predicted_values_gjr_garch = predicted_mean + (conditional_vol_gjr_garch.values[-1, :] ** 0.5) * scale_factor
upper_band_gjr_garch = predicted_mean + 1.96 * (conditional_vol_gjr_garch.values[-1, :] ** 0.5) * scale_factor
lower_band_gjr_garch = predicted_mean - 1.96 * (conditional_vol_gjr_garch.values[-1, :] ** 0.5) * scale_factor

forecast_df_gjr_garch = pd.DataFrame({'Mes': forecast_dates,
                                      'Pronóstico': predicted_values_gjr_garch,
                                      'Inferior 95': lower_band_gjr_garch,
                                      'Superior 95': upper_band_gjr_garch})
forecast_df_gjr_garch.set_index('Mes', inplace=True)

###################################################
################# LSTM ############################
###################################################

pred_lstm, sup_lstm, inf_lstm = lstm_train_with_bands(df['Medio'], forecast_horizon, time_step)
forecast_df_lstm = pd.DataFrame({'Mes': forecast_dates,
                                 'Pronóstico': pred_lstm,
                                 'Inferior 95': inf_lstm,
                                 'Superior 95': sup_lstm})
forecast_df_lstm.set_index('Mes', inplace=True)

###################################################
################# PERCEPTRON ######################
###################################################

pred_nn, sup_nn, inf_nn = nn_train_with_bands(df['Medio'], forecast_horizon, time_step)
forecast_df_nn = pd.DataFrame({'Mes': forecast_dates,
                                 'Pronóstico': pred_nn,
                                 'Inferior 95': inf_nn,
                                 'Superior 95': sup_nn})
forecast_df_nn.set_index('Mes', inplace=True)

###################################################
################# RANDOM FOREST ###################
###################################################

pred_rf, sup_rf, inf_rf = rf_train_with_bands(df['Medio'], forecast_horizon, time_step)
forecast_df_rf = pd.DataFrame({'Mes': forecast_dates,
                                 'Pronóstico': pred_rf,
                                 'Inferior 95': inf_rf,
                                 'Superior 95': sup_rf})
forecast_df_rf.set_index('Mes', inplace=True)


output_path = f'output/pronostico.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    forecast_df_arch.to_excel(writer, sheet_name='ARCH')
    forecast_df_garch.to_excel(writer, sheet_name='GARCH')
    forecast_df_gjr_garch.to_excel(writer, sheet_name='GJR_GARCH')
    forecast_df_lstm.to_excel(writer, sheet_name='LSTM')
    forecast_df_nn.to_excel(writer, sheet_name='NN')
    forecast_df_rf.to_excel(writer, sheet_name='RANDOM_FOREST')
