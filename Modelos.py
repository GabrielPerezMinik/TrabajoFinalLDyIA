from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import PolynomialFeatures,RobustScaler
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
from statsmodels.tsa.statespace.sarimax import SARIMAX

def prophet_train(input_file: str,split_date: str) -> None:
    df = parse_to_date_sales(pd.read_csv(input_file))

    dataframe_plot(df)

    train_series,test_series = split_df(split_date,df)

    df_train_prophet = train_series.reset_index() \
        .rename(columns={'Date':'ds',
                        'Sales':'y'})
    prophet_model = Prophet()
    prophet_model.fit(df_train_prophet)

    df_test_prophet = test_series.reset_index() \
        .rename(columns={'Date':'ds',
                        'Sales':'y'})

    predicted_df = prophet_model.predict(df_test_prophet)
    
    #print(predicted_df)
    
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

    calculate_mse_mae_mape(test_series['Sales'],predicted_df['yhat'])

def arima_train(input_file: str,split_date :str) -> None:
    df = parse_to_date_sales(pd.read_csv(input_file))
    dataframe_plot(df)
    train_series,test_series = split_df(split_date,df)
    model = ARIMA(train_series['Sales'], order=(12, 1, 12)).fit()
    predictions = model.get_forecast(steps=len(test_series)).predicted_mean

    plt.figure(figsize=(10, 6))
    plt.plot(train_series['Date'], train_series['Sales'], label='Train Sales', color='blue')
    plt.plot(test_series['Date'], test_series['Sales'], label='Test Sales', color='green')
    plt.plot(test_series['Date'], predictions, label='Predicted Sales', color='red')
    plt.title('ARIMA Model: Train, Test, and Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    calculate_mse_mae_mape(test_series['Sales'],predictions)

def sarimax_train(input_file: str,split_date :str) -> None:
    df = parse_to_date_sales(pd.read_csv(input_file))
    dataframe_plot(df)
    train_series,test_series = split_df(split_date,df)
    model = SARIMAX(train_series['Sales'], order=(12, 1, 12)).fit()
    predictions = model.get_forecast(steps=len(test_series)).predicted_mean

    plt.figure(figsize=(10, 6))
    plt.plot(train_series['Date'], train_series['Sales'], label='Train Sales', color='blue')
    plt.plot(test_series['Date'], test_series['Sales'], label='Test Sales', color='green')
    plt.plot(test_series['Date'], predictions, label='Predicted Sales', color='red')
    plt.title('SARIMAX Model: Train, Test, and Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    calculate_mse_mae_mape(test_series['Sales'],predictions)

def calculate_mse_mae_mape(expected_value,predictions) -> None:
    
    mape = mean_absolute_percentage_error(expected_value, predictions)
    mse = np.sqrt(mean_squared_error(y_true=expected_value,y_pred=predictions))
    mae = mean_absolute_error(y_true=expected_value,y_pred=predictions)

    print("\n")
    print(f"Mean Squared Error: {mse:.2f}")
    
    print(f"Mean Absolute Error: {mae:.2f}")

    print(f"Mean Absolute Percentage Error (Only relevant without RobustScaling): {mape:.2f}%")

def dataframe_plot(df: pd.DataFrame) -> None:
        ax = df.plot(x='Date',y='Sales',
            style='-',
            figsize=(10, 5),
            ms=1)
        ax.legend()
        ax.set_title("Dataframe Values")
        plt.show()

def split_df(split_date,df) -> pd.DataFrame:
    split_date = pd.to_datetime(split_date)
    df['Date'] = pd.to_datetime(df['Date'])

    first_df = df[(df['Date'] <= split_date)]
    second_df = df[(df['Date'] >= split_date)]

    first_df['Date'] = pd.to_datetime(first_df['Date'])

    second_df['Date'] = pd.to_datetime(second_df['Date'])


    plt.figure(figsize=(10, 5))
    plt.plot(first_df['Date'],first_df['Sales'], label="Training Data")
    plt.plot(second_df['Date'],second_df['Sales'], label="Validation Data")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Data Distribution")
    plt.legend()
    plt.show()
    return first_df,second_df

def mean_absolute_percentage_error(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def parse_to_date_sales(df: pd.DataFrame) -> pd.DataFrame:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Date'] = df['InvoiceDate'].dt.date
    df = df.groupby('Date').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum()).reset_index(name='Sales').round(2)
    
    scaler = RobustScaler()

    df['Sales'] = scaler.fit_transform(df[['Sales']])
    return df

def print_menu() -> None:
    print("1. Prophet")
    print("2. Arima")
    print("3. Sarimax")
    print("4. Exit")

#MAIN
input_file = './data/clean_data.csv'
split_date = '2011-11-08'

option = 0
while(True):
    print_menu()
    option = int(input("User Input: "))
    if(option == 1):
        print("Now training Prophet...")
        prophet_train(input_file, split_date)
    elif (option == 2):
        print("Now training Arima...")
        arima_train(input_file, split_date)
    elif (option == 3):
        print("Now training Sarimax...")
        sarimax_train(input_file, split_date)
    elif (option == 4):
        print("Exitting...")
        break
    else:
        print("Unkown input \n")