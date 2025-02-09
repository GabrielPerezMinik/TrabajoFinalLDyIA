import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
#import cupy as cp

def print_results(df) -> None:
    print("\nHeader:")
    print(df.head())
    print("\nDescribe:")
    print(df.describe())
    print("\nNulls:")
    print(df.isnull().sum())
    negatives_values = df[(df['Quantity'] < 0) | (df['UnitPrice'] < 0)]
    print(f"\nTotal number of negatives: {negatives_values.__len__()} rows")

def outliers_by_quartiles(df) -> pd.DataFrame:
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

def outliers_by_threshold(threshold,df) -> pd.DataFrame:
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df


input_file = "./data/data.csv"
output_file = "./data/clean_data.csv"
numeric_cols = ['Quantity', 'UnitPrice']

df = pd.read_csv(input_file, encoding="latin1") #Change encoding for working purposes
new_df = df

new_df = new_df[(new_df['Quantity'] >= 0) & (new_df['UnitPrice'] >= 0)]
print_results(df)

new_df = outliers_by_threshold(20,new_df)

print_results(new_df)
new_df.to_csv(output_file, index=False)
