from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#aqui te mostrara la GPU si esta configurada para usar la libreria Cupy
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

#verificar que LSTM esta instalado
print(tf.keras.layers.LSTM())

print("hello world")