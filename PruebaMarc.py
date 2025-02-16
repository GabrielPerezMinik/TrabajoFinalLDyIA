import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
import re
#import cupy as cp

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, LabelEncoder, MinMaxScaler, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from datetime import datetime, timedelta
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.decomposition import PCA
from skforecast.recursive import ForecasterRecursive
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

#Indicamos la ruta del archivo CSV
file_path = "data/data.csv"

#Cargamos el DATASET
df = pd.read_csv(file_path, encoding = "latin1")
#Visualizamos las primeras filas
print(df.head(10))
df.info()
df.describe()

groups = df.groupby(['InvoiceNo', 'Description'])

count_groups = groups.size()

actual_duplicates = count_groups[count_groups > 1].reset_index()

duplicate_rows = pd.merge(actual_duplicates, df, on=['InvoiceNo', 'Description'], how='left')
duplicate_rows.head(10)

null_values_before = df.isnull().sum()
print(f"These are the null values: {null_values_before}")

df.isnull().sum().plot(ylim=(0, 140000), kind='bar',color='midnightblue', edgecolor='white')
rows_with_null_description = df[df["Description"].isnull()]

also_with_null_customer_id  = rows_with_null_description["CustomerID"].isnull().all()

print(f"Rows with null description also have null customerID:  {also_with_null_customer_id }")

print(f"Rows before removing null Descriptions: {df.shape[0]}")

df = df.dropna(subset=["Description"])

print(f"Rows after removing null Descriptions: {df.shape[0]}")


#Negative values

negative_values_unit_price = df[df["UnitPrice"] < 0]
print(f"Negative values in the UnitPrice column:\n {negative_values_unit_price}")
negative_values_qauntity = df[df["Quantity"] < 0]
print(f"\nNegative values in the Quantity column:\n {negative_values_qauntity}")


df = df[(df["Quantity"] >=0) & (df["UnitPrice"] >=0)]

plt.figure(figsize=(8, 6))
sns.boxplot(y=df['UnitPrice'])
plt.title(' UnitPrice box plot')
plt.ylabel('UnitPrice')
plt.show()

top_25_high_values = df["UnitPrice"].nlargest(30)

rows_25_high = df[df["UnitPrice"].isin(top_25_high_values)]

useless_descriptions = re.compile(r"(Manual|DOTCOM POSTAGE|AMAZON FEE|POSTAGE|Adjust bad debt)", re.IGNORECASE)

#We remove these rows

df = df[~df["Description"].str.contains(useless_descriptions, na=False)]

# Check the number of rows after deletion
print(f"Number of rows after deletion: {df.shape[0]}")

#We see the graph of possible outliers after deleting the rows we have found

# Box plot of UnitPrice
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['UnitPrice'])
plt.title('Box Plot of UnitPrice')
plt.ylabel('UnitPrice')
plt.show()

#We show the 25 highest values ​​again to understand if we have any more values ​​to eliminate

top_25_high_values = df["UnitPrice"].nlargest(10)

rows_25_high = df[df["UnitPrice"].isin(top_25_high_values)]

rows_25_high

#We see that now all the descriptions would be correct
#We do the same with Quantity
# UnitPrice box plot
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['Quantity'])
plt.title('Quantity Box Plot')
plt.ylabel('UnitPrice')
plt.show()

#We show the 25 highest values ​​again to understand if we have any more values ​​to eliminate

top_25_high_values = df["Quantity"].nlargest(10)

rows_25_high = df[df["Quantity"].isin(top_25_high_values)]

rows_25_high

#We see that with the same process we cannot draw any conclusions, so in this case we will do it with the standard deviation.

#Calculate the standard deviation
mean_quantity = df["Quantity"].mean()
standard_deviation_quantity = df["Quantity"].std()

threshold_extremes = 20 * standard_deviation_quantity

#Filter the dataset to show the outliers

outliers_standard_deviation = df[(df["Quantity"] < mean_quantity - threshold_extremes) | (df["Quantity"] > mean_quantity + threshold_extremes)]
print(outliers_standard_deviation)


print(f"Number of rows before deletion: {df.shape[0]}")

#We eliminate the values ​​that we have considered outliers
df = df[~df["Quantity"].isin(outliers_standard_deviation["Quantity"])]

# Check the number of rows after deletion
print(f"Number of rows after deletion: {df.shape[0]}")

# Filter rows with UnitPrice equal to 0
filas_unitprice_cero = df[df['UnitPrice'] == 0]

# Print the number of rows
print(f"Number of rows with UnitPrice equal to 0: {len(filas_unitprice_cero)}")

# Print the rows
print(filas_unitprice_cero)

# Filter rows with UnitPrice equal to 0, CustomerID null
filas_problematicas = df[(df['UnitPrice'] == 0) | (df['CustomerID'].isnull())]

# Print the number of rows
print(f"Number of rows with UnitPrice equal to 0, CustomerID null and Quantity negative: {len(filas_problematicas)}")

# Print the rows (optional)
print(filas_problematicas)
print(f"Number of rows before deletion: {df.shape[0]}")

# Remove problematic rows from the DataFrame
df = df[(df['UnitPrice'] != 0) & (df['CustomerID'].notna())]

# Check the number of rows after deletion
print(f"Number of rows after deletion: {df.shape[0]}")

#UNIQUE VALUES
#View number of unique values
number_of_unique_values = df["CustomerID"].nunique()
print(f"The number of unique values in 'Quantity is: {number_of_unique_values}")

# Crear una figura
plt.figure(figsize=(10, 6))

# Graficos paises que más compran
plt.subplot(1, 2, 1)
plt.title("Gráfico para los 3 páises que más compran")
ciudades = df["Country"].value_counts().sort_values(ascending=False)[:3]
plt.pie(ciudades.values, labels=ciudades.index, autopct="%1.1f%%")

plt.tight_layout()

plt.subplot(1, 2, 2)
plt.title("Gráfico para los 4-10 páises que más compran")
ciudades = df["Country"].value_counts().sort_values(ascending=False)[3:10]
plt.pie(ciudades.values, labels=ciudades.index, autopct="%1.1f%%")

# Mostrar el gráfico
plt.show()

# Create the 'TotalPrice' column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Convert the 'InvoiceDate' column to datetime BEFORE groupby
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create the daily variables
df_daily = df.groupby('InvoiceDate').agg(
Daily_sales=('Quantity', lambda x: x[x > 0].sum()),
Daily_returns=('Quantity', lambda x: abs(x[x < 0].sum())), # Apply abs() to get the absolute value
Transactions_per_day=('InvoiceNo', 'nunique'),
Daily_income=('TotalPrice', lambda x: x[x > 0].sum()),
Daily_credits=('TotalPrice', lambda x: x[x < 0].sum()),
Daily_profit=('TotalPrice', 'sum'),
Weekday=('InvoiceDate', lambda x: x.iloc[0].dayofweek)

)

# Adjust the total transactions
df_daily['Transactions_per_day'] = df_daily['Daily_sales'] + df_daily['Daily_returns']

# Print the resulting DataFrame
print(df_daily)

encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[["Country"]])

pca = PCA()

pca_dimensionality = pca.fit_transform(df[["InvoiceNo", "Quantity", "UnitPrice", "CustomerID"]])

# Convert 'InvoiceDate' column to datetime
df_daily.index = pd.to_datetime(df_daily.index)

# Filter data between December 1, 2010 and November 8, 2011
data_train_val = df_daily[(df_daily.index >= datetime(2010, 12, 1)) & (df_daily.index <= datetime(2011, 11, 8))]

# Split data into training and validation sets (80% and 20%)
df_daily_train, df_daily_val = train_test_split(data_train_val, test_size=0.2, random_state=42) # random_state for reproducibility

# Filter data for test set (after November 8, 2011) 2011)
df_daily_test = df_daily[(df_daily.index >= datetime(2011, 11, 9)) & (df_daily.index <= datetime(2011, 12, 9))]

# Print the size of the sets
print(f"Training set size: {df_daily_train.shape[0]}")
print(f"Validation set size: {df_daily_val.shape[0]}")
print(f"Test set size: {df_daily_test.shape[0]}")
# Columns to normalize
cols_to_scale = ['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Daily_profit', 'Weekday']

# Instantiate the scalers
robust_scaler = RobustScaler()

# Fit the scalers to the training data
robust_scaler.fit(df_daily_train[cols_to_scale])

# Transform the training, validation, and test data with RobustScaler
df_daily_train_robust = df_daily_train.copy()
df_daily_val_robust = df_daily_val.copy()
df_daily_test_robust = df_daily_test.copy()
df_daily_train_robust[cols_to_scale] = robust_scaler.transform(df_daily_train_robust[cols_to_scale])
df_daily_val_robust[cols_to_scale] = robust_scaler.transform(df_daily_val_robust[cols_to_scale])
df_daily_test_robust[cols_to_scale] = robust_scaler.transform(df_daily_test_robust[cols_to_scale])

#Random Forest Model
# Function to evaluate the model
def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions) #, squared=False
    return rmse

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

param_grid_poly = {
    'polynomialfeatures__degree': [2, 3], # Degree of the polynomial
    'linearregression__fit_intercept': [True, False] # Fit the intercept
}

# Create a list for store the results
results = []

# Iterate over the different scalers
for scaler, train_data, val_data, test_data in [
    (RobustScaler(), df_daily_train_robust, df_daily_val_robust, df_daily_test_robust)
]:
    # Create an instance of the model
    model = RandomForestRegressor(random_state=42)

    # Fit the GridSearchCV to the training and validation data
    grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    )
    grid_search. fit(
    pd. concat([train_data, val_data])[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']],
    pd. concat([train_data, val_data])['Daily_profit']
    )

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    rmse_test = evaluate_model(
    best_model,
    pd. concat([train_data, val_data])[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']],
    pd. concat([train_data, val_data])['Daily_profit'],
    test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']],
    test_data['Daily_profit']
    )

    polynomial_model = make_pipeline(
    scaler, # To normalize the features
    PolynomialFeatures(), # Polynomial transformation
    LinearRegression() # Linear regression model
    )

    # Fit the Polynomial Regression GridSearchCV to the training and validation data
    grid_search_poly = GridSearchCV(
    estimator=polynomial_model,
    param_grid=param_grid_poly,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    )
    grid_search_poly.fit(
    pd.concat([train_data, val_data])[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']],
    pd. concat([train_data, val_data])['Daily_profit']
    )

    # Get the best polynomial regression model
    best_poly_model = grid_search_poly.best_estimator_

    # Evaluate the best model on the test set
    rmse_test_poly = evaluate_model(
    best_poly_model,
    pd. concat([train_data, val_data])[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']],
    pd. concat([train_data, val_data])['Daily_profit'],
    test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']],
    test_data['Daily_profit']
    )

    # Save the results
    results.append({
    'model': 'Random Forest',
    'rmse_test': rmse_test,
    'best_parameters': grid_search.best_params_,
    })

    results.append({
    'model': 'Polynomial Regression',
    'rmse_test': rmse_test_poly,
    'best_parameters': grid_search_poly.best_params_,
    })

# Print the results
for result in results:
    print(f"Results with {result['model']}:")
    print(f" RMSE on test set: {result['rmse_test']}")
    print(f" Best parameters: {result['best_parameters']}")

    # Perform predictions on the test set for both models
    predictions_rf = best_model.predict(
        test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']])
    predictions_poly = best_poly_model.predict(
        test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']])

    # Obtener los valores reales del conjunto de prueba
    y_test = test_data['Daily_profit']

    # Crear el gráfico de dispersión
    plt.figure(figsize=(12, 6))

    # Random Forest predictions
    plt.scatter(y_test, predictions_rf, alpha=0.6, label='Random Forest', color='blue')

    # Polynomial Regression predictions
    plt.scatter(y_test, predictions_poly, alpha=0.6, label='Polynomial Regression', color='green')

    # Add a diagonal reference line to compare the predictions with the actual values
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Línea Ideal')

    # Titles and labels
    plt.title('Comparison of predictions with actual values', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predictions', fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculate the errors (differences between predictions and actual values)
    errors_rf = predictions_rf - y_test
    errors_poly = predictions_poly - y_test

    # Create the error plot
    plt.figure(figsize=(12, 6))

    # Plot the Random Forest errors
    plt.scatter(y_test, errors_rf, alpha=0.6, label='Errores Random Forest', color='blue')

    # Plot the Polynomial Regression errors
    plt.scatter(y_test, errors_poly, alpha=0.6, label='Errores Polynomial Regression', color='green')

    # Titles and labels
    plt.title('Prediction Errors Compared to Actual Values', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculate the mean absolute error (MAE) for each model
    mae_rf = abs(errors_rf).mean()
    mae_poly = abs(errors_poly).mean()

    # Create the bar chart to compare the mean error
    plt.figure(figsize=(8, 6))

    # Create the bars for each model
    plt.bar(['Random Forest', 'Polynomial Regression'], [mae_rf, mae_poly], color=['blue', 'green'])

    # Titles and labels
    plt.title('Comparison of Mean Error by Model', fontsize=16)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Mean Error (MAE)', fontsize=14)

    plt.tight_layout()
    plt.show()
