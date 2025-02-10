import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from datetime import datetime
from sklearn.pipeline import make_pipeline
import re

# Función para cargar los datos
def load_data(file_path):
    return pd.read_csv(file_path, encoding="latin1")

# Función para limpiar los datos
def clean_data(df):
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Eliminar descripciones nulas
    df = df.dropna(subset=["Description"])

    # Eliminar valores negativos de Quantity y UnitPrice
    df = df[(df["Quantity"] >= 0) & (df["UnitPrice"] >= 0)]

    # Eliminar valores de descripción no útiles
    useless_descriptions = re.compile(r"(Manual|DOTCOM POSTAGE|AMAZON FEE|POSTAGE|Adjust bad debt)", re.IGNORECASE)
    df = df[~df["Description"].str.contains(useless_descriptions, na=False)]

    return df

# Función para procesar las columnas necesarias
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

    # Ajustar el total de transacciones
    df_daily['Transactions_per_day'] = df_daily['Daily_sales'] + df_daily['Daily_returns']
    return df_daily

# Función para dividir los datos en entrenamiento, validación y prueba
def split_data(df_daily):
    data_train_val = df_daily[(df_daily.index >= datetime(2010, 12, 1)) & (df_daily.index <= datetime(2011, 11, 8))]
    df_daily_train, df_daily_val = train_test_split(data_train_val, test_size=0.2, random_state=42)
    
    df_daily_test = df_daily[(df_daily.index >= datetime(2011, 11, 9)) & (df_daily.index <= datetime(2011, 12, 9))]
    
    return df_daily_train, df_daily_val, df_daily_test

# Función para normalizar los datos
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

# Función para entrenar y evaluar los modelos
def train_and_evaluate_models(df_daily_train_robust, df_daily_val_robust, df_daily_test_robust):
    results = []

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
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
            pd.concat([train_data, val_data])[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']],
            pd.concat([train_data, val_data])['Daily_profit']
        )

        best_model = grid_search.best_estimator_

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
            pd.concat([train_data, val_data])[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']],
            pd.concat([train_data, val_data])['Daily_profit']
        )

        best_poly_model = grid_search_poly.best_estimator_

        # Guardar los resultados
        results.append({
            'model': 'Random Forest',
            'rmse_test': evaluate_model(best_model, pd.concat([train_data, val_data]), test_data),
            'best_parameters': grid_search.best_params_,
        })

        results.append({
            'model': 'Polynomial Regression',
            'rmse_test': evaluate_model(best_poly_model, pd.concat([train_data, val_data]), test_data),
            'best_parameters': grid_search_poly.best_params_,
        })

        # Crear gráficos de comparación
        create_comparison_graphs(best_model, best_poly_model, test_data)

    return results

# Función para evaluar el modelo
def evaluate_model(model, train_val_data, test_data):
    X_train_val = train_val_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']]
    y_train_val = train_val_data['Daily_profit']
    X_test = test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']]
    y_test = test_data['Daily_profit']
    
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions)
    return rmse

# Función para crear los gráficos de comparación
def create_comparison_graphs(best_model, best_poly_model, test_data):
    predictions_rf = best_model.predict(test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']])
    predictions_poly = best_poly_model.predict(test_data[['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Weekday']])

    # Obtener los valores reales del conjunto de prueba
    y_test = test_data['Daily_profit']

    # Gráfico de dispersión comparando las predicciones con los valores reales
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

    # Calcular los errores
    errors_rf = predictions_rf - y_test
    errors_poly = predictions_poly - y_test

    # Gráfico de dispersión de los errores
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, errors_rf, alpha=0.6, label='Errors Random Forest', color='blue')
    plt.scatter(y_test, errors_poly, alpha=0.6, label='Errors Polynomial Regression', color='green')
    plt.title('Prediction Errors Compared to Actual Values', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calcular el error medio absoluto (MAE) para cada modelo
    mae_rf = abs(errors_rf).mean()
    mae_poly = abs(errors_poly).mean()

    # Gráfico de barras comparando el error medio
    plt.figure(figsize=(8, 6))
    plt.bar(['Random Forest', 'Polynomial Regression'], [mae_rf, mae_poly], color=['blue', 'green'])
    plt.title('Comparison of Mean Error by Model', fontsize=16)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Mean Error (MAE)', fontsize=14)
    plt.tight_layout()
    plt.show()

# Función para mostrar los resultados
def print_results(results):
    for result in results:
        print(f"Results with {result['model']}:")
        print(f" RMSE on test set: {result['rmse_test']}")
        print(f" Best parameters: {result['best_parameters']}")

# Función principal para ejecutar el flujo completo
def main():
    file_path = './data.csv'
    
    # Cargar datos
    df = load_data(file_path)
    
    # Limpiar datos
    df = clean_data(df)
    
    # Procesar columnas
    df_daily = process_columns(df)
    
    # Dividir datos
    df_daily_train, df_daily_val, df_daily_test = split_data(df_daily)
    
    # Normalizar datos
    cols_to_scale = ['Daily_sales', 'Daily_returns', 'Transactions_per_day', 'Daily_income', 'Daily_credits', 'Daily_profit', 'Weekday']
    df_daily_train_robust, df_daily_val_robust, df_daily_test_robust = normalize_data(df_daily_train, df_daily_val, df_daily_test, cols_to_scale)
    
    # Entrenar y evaluar modelos
    results = train_and_evaluate_models(df_daily_train_robust, df_daily_val_robust, df_daily_test_robust)
    
    # Mostrar resultados
    print_results(results)

# Ejecutar el flujo principal
if __name__ == "__main__":
    main()