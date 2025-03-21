from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
#<<<<<<< HEAD
#=======
from sklearn.preprocessing import PolynomialFeatures,RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
#>>>>>>> master
from xgboost import XGBRegressor
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from sklearn.pipeline import make_pipeline
import re
from sklearn.model_selection import cross_val_score


# Function to load the cleaned data
def load_cleaned_data(file_path):
    """Load the CSV file with the cleaned data."""
    return pd.read_csv(file_path, encoding="latin1")


# Function to clean the data
def clean_data(df):
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Remove null descriptions
    df = df.dropna(subset=["Description"])

    # Remove negative values ​​from Quantity and UnitPrice
    df = df[(df["Quantity"] >= 0) & (df["UnitPrice"] >= 0)]

    # Remove unhelpful descriptions
    useless_descriptions = re.compile(r"(Manual|DOTCOM POSTAGE|AMAZON FEE|POSTAGE|Adjust bad debt)", re.IGNORECASE)
    df = df[~df["Description"].str.contains(useless_descriptions, na=False)]

    return df



# Function to process the necessary columns
def process_columns(df):
    df_daily = df.groupby('InvoiceDate').agg(
        Daily_sales=('Quantity', lambda x: x[x > 0].sum()),
        Daily_returns=('Quantity', lambda x: abs(x[x < 0].sum())),  # Apply abs() to get the absolute value
        Transactions_per_day=('InvoiceNo', 'nunique'),
        Daily_income=('TotalPrice', lambda x: x[x > 0].sum()),
        Daily_credits=('TotalPrice', lambda x: x[x < 0].sum()),
        Daily_profit=('TotalPrice', 'sum'),
        Weekday=('InvoiceDate', lambda x: x.iloc[0].dayofweek)
    )

   # Adjust total transactions
    df_daily['Transactions_per_day'] = df_daily['Daily_sales'] + df_daily['Daily_returns']
    return df_daily

# Function to split data into training, validation and testing
def split_data(df_daily):
    # Entrenamiento y validación: De diciembre 2010 a noviembre 2011
    data_train_val = df_daily[(df_daily.index >= datetime(2010, 12, 1)) & (df_daily.index <= datetime(2011, 11, 8))]

    # 80% de los datos para entrenamiento, 20% para validación (sin aleatorización)
    df_daily_train = data_train_val.iloc[:int(len(data_train_val) * 0.8)]
    df_daily_val = data_train_val.iloc[int(len(data_train_val) * 0.8):]

    # Conjunto de prueba: De noviembre 9 a diciembre 9 de 2011
    df_daily_test = df_daily[(df_daily.index >= datetime(2011, 11, 9)) & (df_daily.index <= datetime(2011, 12, 9))]

    return df_daily_train, df_daily_val, df_daily_test

# Function to normalize data
def normalize_data(df_daily_train, df_daily_val, df_daily_test, cols_to_scale):
    '''robust_scaler = RobustScaler()
    robust_scaler.fit(df_daily_train[cols_to_scale])
    
    df_daily_train_robust = df_daily_train.copy()
    df_daily_val_robust = df_daily_val.copy()
    df_daily_test_robust = df_daily_test.copy()
    
    df_daily_train_robust[cols_to_scale] = robust_scaler.transform(df_daily_train_robust[cols_to_scale])
    df_daily_val_robust[cols_to_scale] = robust_scaler.transform(df_daily_val_robust[cols_to_scale])
    df_daily_test_robust[cols_to_scale] = robust_scaler.transform(df_daily_test_robust[cols_to_scale])'''
    
    return df_daily_train, df_daily_val, df_daily_test

# Function to train and evaluate the models
def train_and_evaluate_models(df_daily_train_robust, df_daily_val_robust, df_daily_test_robust):
    results = []

    param_grid = {
        'n_estimators': [25, 50],
        'max_depth': [5, 10],
        'min_samples_split': [5, 10]
    }

    param_grid_poly = {
        'polynomialfeatures__degree': [2, 3],  # Degree of the polynomial
        'linearregression__fit_intercept': [True, False]  # Fit the intercept
    }

    for scaler, train_data, val_data, test_data in [
        (RobustScaler(), df_daily_train_robust, df_daily_val_robust, df_daily_test_robust)
    ]:
        # Random Forest Model
        model = RandomForestRegressor(random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_jobs=-1
        )
        
        grid_search.fit(
            train_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']],
            train_data['Daily_profit']
        )

        best_model = grid_search.best_estimator_

        # Evaluar el modelo con los datos de validación antes de usar el conjunto de prueba
        val_predictions = best_model.predict(val_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']])
        val_rmse = mse(val_data['Daily_profit'], val_predictions)
        val_r2 = r2_score(val_data['Daily_profit'], val_predictions)
        val_mae = mean_absolute_error(val_data['Daily_profit'], val_predictions)

         # Evaluate the model with cross-validation
        cv_results = cross_val_score(best_model, train_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']], 
                                     train_data['Daily_profit'], cv=10, scoring='neg_root_mean_squared_error')
        cv_rmse = np.mean(np.abs(cv_results))
        print(f'Random Forest Model Cross-Validation RMSE: {cv_rmse}')

        # Polynomial Regression
        polynomial_model = make_pipeline(
            scaler, 
            PolynomialFeatures(),
            LinearRegression()
        )

        grid_search_poly = GridSearchCV(
            estimator=polynomial_model,
            param_grid=param_grid_poly,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_jobs=-1
        )

        grid_search_poly.fit(
            train_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']],
            train_data['Daily_profit']
        )

        best_poly_model = grid_search_poly.best_estimator_

        # Evaluar el modelo de regresión polinómica con los datos de validación
        poly_val_predictions = best_poly_model.predict(val_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']])
        poly_val_rmse = mse(val_data['Daily_profit'], poly_val_predictions)
        poly_val_r2 = r2_score(val_data['Daily_profit'], poly_val_predictions)
        poly_val_mae = mean_absolute_error(val_data['Daily_profit'], poly_val_predictions)

        # Evaluate the polynomial model with cross-validation
        cv_poly_results = cross_val_score(best_poly_model, train_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']], 
                                          train_data['Daily_profit'], cv=10, scoring='neg_root_mean_squared_error')
        cv_poly_rmse = np.mean(np.abs(cv_poly_results))
        print(f'Polynomial Regression Cross-Validation RMSE: {cv_poly_rmse}')

        # Save the results
        results.append({
            'model': 'Random Forest',
            'rmse_test': evaluate_model(best_model, train_data, test_data),
            'r2_test': evaluate_model(best_model, train_data, test_data),
            'mae_test': evaluate_model(best_model, train_data, test_data),
            'best_parameters': grid_search.best_params_,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae,
            'cv_rmse': cv_rmse
        })

        results.append({
            'model': 'Polynomial Regression',
            'rmse_test': evaluate_model(best_poly_model, train_data, test_data),
            'r2_test': evaluate_model(best_poly_model, train_data, test_data),
            'mae_test': evaluate_model(best_poly_model, train_data, test_data),
            'best_parameters': grid_search_poly.best_params_,
            'val_rmse': poly_val_rmse,
            'val_r2': poly_val_r2,
            'val_mae': poly_val_mae,
            'cv_rmse': cv_poly_rmse
        })

        # Create comparison charts
        create_comparison_graphs(best_model, best_poly_model, test_data)

    return results

# Function to evaluate the model
def evaluate_model(model, train_val_data, test_data):
    X_train_val = train_val_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']]
    y_train_val = train_val_data['Daily_profit']
    X_test = test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']]
    y_test = test_data['Daily_profit']
    
    predictions = model.predict(X_test)
    rsme = mse(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return {
        'rsme': rsme,
        'r2': r2,
        'mae': mae
    }
    

# Function to create the comparison graphs
def create_comparison_graphs(best_model, best_poly_model, test_data):
    predictions_rf = best_model.predict(test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']])
    predictions_poly = best_poly_model.predict(test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']])

    # Get the actual values ​​of the test set
    y_test = test_data['Daily_profit']

    # Scatter plot comparing predictions to actual values
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, predictions_rf, alpha=0.6, label='Random Forest', color='blue')
    plt.scatter(y_test, predictions_poly, alpha=0.6, label='Polynomial Regression', color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Línea Ideal')
    plt.title('Comparison of predictions with actual values', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predictions', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate the errors
    errors_rf = predictions_rf - y_test
    errors_poly = predictions_poly - y_test

    # Scatter plot of errors
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, errors_rf, alpha=0.6, label='Errors Random Forest', color='blue')
    plt.scatter(y_test, errors_poly, alpha=0.6, label='Errors Polynomial Regression', color='green')
    plt.title('Prediction Errors Compared to Actual Values', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate the mean absolute error (MAE) for each model
    mae_rf = abs(errors_rf).mean()
    mae_poly = abs(errors_poly).mean()

    # Bar chart comparing the mean error
    plt.figure(figsize=(8, 6))
    plt.bar(['Random Forest', 'Polynomial Regression'], [mae_rf, mae_poly], color=['blue', 'green'])
    plt.title('Comparison of Mean Error by Model', fontsize=16)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Mean Error (MAE)', fontsize=14)
    plt.tight_layout()
    plt.show()

# Function to display the results
def print_results(results):
    for result in results:
        print(f"Results with {result['model']}:")
        print(f" RMSE on test set: {result['rmse_test']}")
        print(f" R2 on test set: {result['r2_test']}")
        print(f" MAE on test set: {result['mae_test']}")
        print(f" Best parameters: {result['best_parameters']}")
        print(f" Cross-Validation RMSE: {result['cv_rmse']}")

# Main function to execute the entire flow
def main():
    file_path = 'data/clean_data.csv'
    
    # Load data
    df = load_cleaned_data(file_path)
    
    # Clear data
    df = clean_data(df)
    
    # Process columns
    df_daily = process_columns(df)
    
    # Split data
    df_daily_train, df_daily_val, df_daily_test = split_data(df_daily)
    
    # Normalize data
    cols_to_scale = ['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Daily_profit', 'Weekday']
    df_daily_train_robust, df_daily_val_robust, df_daily_test_robust = normalize_data(df_daily_train, df_daily_val, df_daily_test, cols_to_scale)
    
    # Train and evaluate models
    results = train_and_evaluate_models(df_daily_train_robust, df_daily_val_robust, df_daily_test_robust)
    
    # Show results
    print_results(results)

# Run the main flow
if __name__ == "__main__":
    main()

#<<<<<<< HEAD
#=======
#verificar que LSTM esta instalado
#print(tf.keras.layers.LSTM())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

def prophet_train(input_file: str,split_date: str) -> None:
    """
        Trains Prophet model with given dataframe

        Parameters
        ----------
            input_file (str): full dataframe
            split_date (str): date use to split dataframe into training and validation
    """
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

    calculate_mse_mae_mape("Prophet",test_series['Sales'],predicted_df['yhat'])

def arima_train(input_file: str,split_date :str) -> None:
    """
        Trains Arima model with given dataframe

        Parameters
        ----------
            input_file (str): full dataframe
            split_date (str): date use to split dataframe into training and validation
    """
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

    calculate_mse_mae_mape("Arima",test_series['Sales'],predictions)

def sarimax_train(input_file: str,split_date :str) -> None:
    """
        Trains Sarimax model with given dataframe

        Parameters
        ----------
            input_file (str): full dataframe
            split_date (str): date use to split dataframe into training and validation
    """
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

    calculate_mse_mae_mape("Sarimax",test_series['Sales'],predictions)

def calculate_mse_mae_mape(model_name: str,expected_value,predictions) -> None:
    """
        Displays mse mae and mape

        Parameters
        ----------
            model_name (str): Model trained
            expected_value (any arraylike): Real value from the dataframe
            predictions (any arraylike): Predicted value from the model
    """
    mape = mean_absolute_percentage_error(expected_value, predictions)
    mse = np.sqrt(mean_squared_error(y_true=expected_value,y_pred=predictions))
    mae = mean_absolute_error(y_true=expected_value,y_pred=predictions)

    print("\n")
    print(f"Modelo entrenado: {model_name}")

    print(f"Mean Squared Error: {mse:.2f}")
    
    print(f"Mean Absolute Error: {mae:.2f}")

    print(f"Mean Absolute Percentage Error (Only relevant without RobustScaling): {mape:.2f}%")

   



def dataframe_plot(df: pd.DataFrame) -> None:
        """
        Displays dataframe info

        Parameters
        ----------
            df (pd.dataframe): dataframe to display
        """
        ax = df.plot(x='Date',y='Sales',
            style='-',
            figsize=(10, 5),
            ms=1)
        ax.legend()
        ax.set_title("Dataframe Values")
        plt.show()

def split_df(split_date: str,df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits dataframe by given date

    Parameters
    ----------
        split_date (str): date to split, format: yyy-mm-dd
        df (pd.dataframe): dataframe to split

    Returns:
        dataframe (pd.Dataframe): dataframe up to the split date
        dataframe (pd.Dataframe): dataframe from the split date to end
    """
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

def mean_absolute_percentage_error(expected, predicted) -> float:
    """
    calculates mape

    Parameters
    ----------
        expected (array-like): Real value from the dataframe
        predicted (array-like): Predicted value from the model

    Returns:
        error percentage (float): error percentage
    """
    expected, predicted = np.array(expected), np.array(predicted)
    return np.mean(np.abs((expected - predicted) / expected)) * 100

def parse_to_date_sales(df: pd.DataFrame) -> pd.DataFrame:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Date'] = df['InvoiceDate'].dt.date
    df = df.groupby('Date').apply(lambda x: (x['Quantity'] * x['UnitPrice']).sum()).reset_index(name='Sales').round(2)
    
    scaler = RobustScaler()

    df['Sales'] = scaler.fit_transform(df[['Sales']])
    return df


def print_menu() -> None:
    """
    Shows the main menu
    """
    print("1. Prophet")
    print("2. Arima")
    print("3. Sarimax")
    print("4. Exit")

#MAIN
input_file = './data/clean_data.csv' # File for training
split_date = '2011-11-08' # Date for splitting file

while(True):
    print_menu()
    match int(input("User Input: ")):
        case 1:
            print("Now training Prophet...")
            prophet_train(input_file, split_date)
        case 2:
            print("Now training Arima...")
            arima_train(input_file, split_date)
        case 3:
            print("Now training Sarimax...")
            sarimax_train(input_file, split_date)
        case 4:
            print("Exitting...")
            break
        case _:
            print("Unkown input \n")
#>>>>>>> master
