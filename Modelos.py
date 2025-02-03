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
#print(tf.keras.layers.LSTM())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error

def prophet_train():
    df = parse_to_date_sales(pd.read_csv('./data/clean_data.csv'))

    print(df.columns)
    ax = df.plot(x='Date',y='Sales',
        style='-',
        figsize=(10, 5),
        ms=1)
    ax.legend()
    ax.set_title("Dataframe Values")
    plt.show()

    train_series,test_series = split_df('2011-11-08',df)

    df_train_prophet = train_series.reset_index() \
        .rename(columns={'Date':'ds',
                        'Sales':'y'})
    print(df_train_prophet)
    prophet_model = Prophet()
    prophet_model.fit(df_train_prophet)

    df_test_prophet = test_series.reset_index() \
        .rename(columns={'Date':'ds',
                        'Sales':'y'})

    predicted_df = prophet_model.predict(df_test_prophet)
    
    print(predicted_df)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    prophet_model.plot(predicted_df, ax=ax)
    ax.legend()
    ax.set_title("Dataframe / Predicted")
    plt.show()

    prophet_model.plot_components(predicted_df)
    plt.show()

    f, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(test_series.Date, test_series['Sales'], color='r')
    fig = prophet_model.plot(predicted_df, ax=ax)
    ax.legend()
    ax.set_title("Real dataframe and Predicted Dataframe compared")
    plt.show()

    f, ax = plt.subplots(figsize=(10, 5))

    ax.plot(test_series['Date'], test_series['Sales'], color='green', label='Valores Reales')

    fig = prophet_model.plot(predicted_df, ax=ax)

    ax.set_title('Real dataframe and Predicted Dataframe compared')
    ax.legend()
    plt.show()

    print("Mean Squared Error: ")
    print(np.sqrt(mean_squared_error(y_true=test_series['Sales'],
                   y_pred=predicted_df['yhat'])))
    
    print("\n")
    print("Mean Absolute Error: ")
    print(mean_absolute_error(y_true=test_series['Sales'],
                   y_pred=predicted_df['yhat']))
    print("\n")

    print("Mean Absolute Percentage Error: ")
    print(mean_absolute_percentage_error(y_true=test_series['Sales'],
                   y_pred=predicted_df['yhat']))

def split_df(split_date,df):
    split_date = pd.to_datetime(split_date)
    df['Date'] = pd.to_datetime(df['Date'])

    first_df = df[(df['Date'] <= split_date)]
    second_df = df[(df['Date'] >= split_date)]

    first_df['Date'] = pd.to_datetime(first_df['Date'])

    second_df['Date'] = pd.to_datetime(second_df['Date'])


    plt.figure(figsize=(10, 5))
    plt.plot(first_df, label="Training Data")
    plt.plot(second_df, label="Validation Data")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Data Distribution")
    plt.legend()
    plt.show()
    return first_df,second_df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def parse_to_date_sales(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Date'] = df['InvoiceDate'].dt.date
    df = df.groupby('Date').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum()).reset_index(name='Sales').round(2)
    return df

#MAIN
prophet_train()