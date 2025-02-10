import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import re


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_data(file_path):
    """Loads dataset from a CSV file."""
    return pd.read_csv(file_path, encoding="latin1")

def display_head_and_info(df):
    """Displays the first 10 rows, info, and description of the dataframe."""
    print(df.head(10))
    df.info()
    df.describe()

def group_and_find_duplicates(df):
    """Finds and returns duplicate rows based on InvoiceNo and Description."""
    groups = df.groupby(['InvoiceNo', 'Description'])
    count_groups = groups.size()
    actual_duplicates = count_groups[count_groups > 1].reset_index()
    duplicate_rows = pd.merge(actual_duplicates, df, on=['InvoiceNo', 'Description'], how='left')
    return duplicate_rows

def handle_null_values(df):
    """Handles null values in the dataset."""
    null_values_before = df.isnull().sum()
    print(f"These are the null values: {null_values_before}")
    df.isnull().sum().plot(ylim=(0, 140000), kind='bar', color='midnightblue', edgecolor='white')
    
    rows_with_null_description = df[df["Description"].isnull()]
    also_with_null_customer_id = rows_with_null_description["CustomerID"].isnull().all()
    print(f"Rows with null description also have null customerID: {also_with_null_customer_id}")

    # Remove rows with null Description
    print(f"Rows before removing null Descriptions: {df.shape[0]}")
    df = df.dropna(subset=["Description"])
    print(f"Rows after removing null Descriptions: {df.shape[0]}")
    
    return df

def handle_negative_values(df):
    """Identifies and removes negative values in Quantity and UnitPrice."""
    negative_values_unit_price = df[df["UnitPrice"] < 0]
    print(f"Negative values in the UnitPrice column:\n {negative_values_unit_price}")
    
    negative_values_quantity = df[df["Quantity"] < 0]
    print(f"\nNegative values in the Quantity column:\n {negative_values_quantity}")

    # Remove negative values
    df = df[(df["Quantity"] >= 0) & (df["UnitPrice"] >= 0)]
    return df

def plot_boxplot(df, column, title, ylabel):
    """Plots a boxplot for a given column."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()

def filter_useless_descriptions(df):
    """Removes rows with useless descriptions like shipping costs or fees."""
    useless_descriptions = re.compile(r"(Manual|DOTCOM POSTAGE|AMAZON FEE|POSTAGE|Adjust bad debt)", re.IGNORECASE)
    df = df[~df["Description"].str.contains(useless_descriptions, na=False)]
    print(f"Number of rows after deletion: {df.shape[0]}")
    return df

def find_and_remove_outliers(df, column, threshold_factor=20):
    """Finds and removes outliers based on standard deviation."""
    mean_value = df[column].mean()
    std_dev_value = df[column].std()
    threshold_extremes = threshold_factor * std_dev_value

    outliers = df[(df[column] < mean_value - threshold_extremes) | (df[column] > mean_value + threshold_extremes)]
    print(outliers)

    # Remove outliers
    print(f"Number of rows before deletion: {df.shape[0]}")
    df = df[~df[column].isin(outliers[column])]
    print(f"Number of rows after deletion: {df.shape[0]}")
    
    return df

def handle_zero_unit_price_and_null_customer_id(df):
    """Handles rows where UnitPrice is zero or CustomerID is null."""
    problematic_rows = df[(df['UnitPrice'] == 0) | (df['CustomerID'].isnull())]
    print(f"Number of rows with UnitPrice equal to 0, CustomerID null: {len(problematic_rows)}")
    print(problematic_rows)
    
    # Remove problematic rows
    print(f"Number of rows before deletion: {df.shape[0]}")
    df = df[(df['UnitPrice'] != 0) & (df['CustomerID'].notna())]
    print(f"Number of rows after deletion: {df.shape[0]}")
    
    return df

def get_unique_values_count(df, column):
    """Returns the number of unique values in a given column."""
    return df[column].nunique()

def plot_country_distribution(df):
    """Plots pie charts for the top countries based on purchases."""
    plt.figure(figsize=(10, 6))

    # Top 3 countries
    plt.subplot(1, 2, 1)
    plt.title("Top 3 Countries by Purchases")
    top_countries = df["Country"].value_counts().sort_values(ascending=False)[:3]
    plt.pie(top_countries.values, labels=top_countries.index, autopct="%1.1f%%")

    # Countries ranked 4th to 10th
    plt.subplot(1, 2, 2)
    plt.title("Countries Ranked 4th to 10th by Purchases")
    other_countries = df["Country"].value_counts().sort_values(ascending=False)[3:10]
    plt.pie(other_countries.values, labels=other_countries.index, autopct="%1.1f%%")

    plt.tight_layout()
    plt.show()

# Main execution
file_path = "./data.csv"
df = load_data(file_path)

display_head_and_info(df)

df = handle_null_values(df)
df = handle_negative_values(df)

# Boxplots
plot_boxplot(df, 'UnitPrice', 'UnitPrice Box Plot', 'UnitPrice')

# Handle outliers and remove outliers in UnitPrice and Quantity
df = filter_useless_descriptions(df)
df = find_and_remove_outliers(df, 'Quantity')

df = handle_zero_unit_price_and_null_customer_id(df)

# Get unique values count in 'CustomerID'
unique_values_count = get_unique_values_count(df, 'CustomerID')
print(f"The number of unique 'CustomerID' values: {unique_values_count}")

# Plot country distribution
plot_country_distribution(df)