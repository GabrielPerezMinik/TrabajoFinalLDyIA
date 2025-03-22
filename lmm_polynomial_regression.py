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
    robust_scaler = RobustScaler()
    robust_scaler.fit(df_daily_train[cols_to_scale])
    
    df_daily_train_robust = df_daily_train.copy()
    df_daily_val_robust = df_daily_val.copy()
    df_daily_test_robust = df_daily_test.copy()
    
    df_daily_train_robust[cols_to_scale] = robust_scaler.transform(df_daily_train_robust[cols_to_scale])
    df_daily_val_robust[cols_to_scale] = robust_scaler.transform(df_daily_val_robust[cols_to_scale])
    df_daily_test_robust[cols_to_scale] = robust_scaler.transform(df_daily_test_robust[cols_to_scale])
    
    return df_daily_train_robust, df_daily_val_robust, df_daily_test_robust

# Function to train and evaluate the models
def train_and_evaluate_models(df_daily_train_robust, df_daily_val_robust, df_daily_test_robust):
    results = []

    param_grid_poly = {
        'polynomialfeatures__degree': [2, 3],  # Degree of the polynomial
        'linearregression__fit_intercept': [True, False]  # Fit the intercept
    }

    for scaler, train_data, val_data, test_data in [
        (RobustScaler(), df_daily_train_robust, df_daily_val_robust, df_daily_test_robust)
    ]:
       
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
        create_comparison_graphs(best_poly_model, test_data)

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
def create_comparison_graphs(best_poly_model, test_data):
    predictions_poly = best_poly_model.predict(test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']])

    # Get the actual values ​​of the test set
    y_test = test_data['Daily_profit']

    # Scatter plot comparing predictions to actual values
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, predictions_poly, alpha=0.6, label='Polynomial Regression', color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Línea Ideal')
    plt.title('Comparison of predictions with actual values', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predictions', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate the errors
    errors_poly = predictions_poly - y_test

    # Scatter plot of errors
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, errors_poly, alpha=0.6, label='Errors Polynomial Regression', color='green')
    plt.title('Prediction Errors Compared to Actual Values', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate the mean absolute error (MAE) for each model
    mae_poly = abs(errors_poly).mean()

    # Bar chart comparing the mean error
    plt.figure(figsize=(8, 6))
    plt.bar(['Polynomial Regression'], [mae_poly], color=['blue', 'green'])
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