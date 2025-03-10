import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

class data_Clean:

    def print_results(df) -> None:
        """
        Prints basic info on given Dataframe

        Parameters
        ----------
            df (pd.Dataframe): Dataframe for printing
        """
        print("\nHeader:")
        print(df.head())
        print("\nDescribe:")
        print(df.describe())
        print("\nNulls:")
        print(df.isnull().sum())
        negatives_values = df[(df['Quantity'] < 0) | (df['UnitPrice'] < 0)]
        print(f"\nTotal number of negatives: {negatives_values.__len__()} rows")

    def outliers_by_quartiles(df,numeric_cols) -> pd.DataFrame:
        """
        Calculate Outliers on given dataframe by quartiles

        Parameters
        ----------
            df (pd.Dataframe): Dataframe to clean.
            numeric_cols (arraylike): all the numeric columns to be used

        Returns:
            dataframe (pd.Dataframe): Dataframe without outliers.
        """
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            return df

    def outliers_by_threshold(threshold,df,numeric_cols) -> pd.DataFrame:
        """
        Calculate Outliers on given dataframe by given threshold

        Parameters
        ----------
            threshold (int): threshold to determinate outliers
            df (pd.Dataframe): Dataframe to clean.
            numeric_cols (arraylike): all the numeric columns to be used
        Returns:
             dataframe (pd.Dataframe): Dataframe without outliers.
        """
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()

            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            return df

def run():
    input_file = "./data/data.csv" # file to parse
    output_file = "./data/clean_data.csv" # path to save parsed file
    numeric_cols = ['Quantity', 'UnitPrice'] # Name of numeric columns in file to calculate outliers

    df = pd.read_csv(input_file, encoding="latin1") # Read file. Change encoding for working purposes
    new_df = df

    new_df = new_df[(new_df['Quantity'] >= 0) & (new_df['UnitPrice'] >= 0)] # Parse null values
    data_Clean.print_results(df)

    new_df = data_Clean.outliers_by_threshold(20,new_df,numeric_cols)

    data_Clean.print_results(new_df)
    new_df.to_csv(output_file, index=False) # Save parsed file
