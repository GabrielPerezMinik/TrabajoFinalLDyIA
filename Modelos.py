from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
#from keras.src.layers import LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/data.csv",encoding="iso-8859-1")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna()

df['profit'] = df['UnitPrice'] * df['Quantity']


# Definir los rangos de fechas
train_start, train_end = "2010-12-01", "2011-11-08"
test_start, test_end = "2011-11-09", "2011-12-09"

# Filtrar los datos
train_data = df[(df['InvoiceDate'] >= train_start) & (df['InvoiceDate'] <= train_end)]
test_data = df[(df['InvoiceDate'] >= test_start) & (df['InvoiceDate'] <= test_end)]



scaler = MinMaxScaler(feature_range=(0, 1))

# Aplicar la normalización
train_scaled = scaler.fit_transform(train_data['profit'].values.reshape(-1, 1))
test_scaled = scaler.transform(test_data['profit'].values.reshape(-1, 1))


def crear_secuencias(data, pasos=30):
    x, y = [], []
    for i in range(pasos, len(data)):
        x.append(data[i-pasos:i, 0])  # Últimos 30 días
        y.append(data[i, 0])  # Día siguiente
    return np.array(x), np.array(y)

# Crear secuencias
x_train, y_train = crear_secuencias(train_scaled, pasos=30)
x_test, y_test = crear_secuencias(test_scaled, pasos=30)

# Ajustar dimensiones para LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Definir el modelo
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(x_train, y_train, epochs=20, batch_size=32)


predicciones = model.predict(x_test)

# Invertir la normalización
predicciones = scaler.inverse_transform(predicciones)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))


plt.figure(figsize=(10,5))
plt.plot(test_data['InvoiceDate'][30:], y_test_real, label='Real')
plt.plot(test_data['InvoiceDate'][30:], predicciones, label='Predicción', linestyle='dashed')
plt.xlabel('date')
plt.ylabel('profit')
plt.title('Prediction of diary profit')
plt.legend()
plt.show()
