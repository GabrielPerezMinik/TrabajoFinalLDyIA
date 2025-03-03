import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def optimise_k_means(df,max_k):
    """
    Method used to find a break point on the number of clusters to use
    
    Parameters
    ----------
        df (pd.dataframe): dataframe for training
        max_k (int): number of max clusters or iterations method will go through
    """
    print("Calculating k-means Clusters...")
    means = []
    iterations = []
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        means.append(k)
        iterations.append(kmeans.inertia_)
    
    fig = plt.subplot()
    plt.plot(means, iterations, 'o-')
    plt.grid(True)
    plt.xlabel("Clusters")
    plt.ylabel("Iterations")
    plt.show()

def kmeans_cluster_by_encoding(df: pd.DataFrame, k: int, additional_info: bool) -> None:
    """
    Trains Kmeans model

    Parameters
    ----------
        df (pd.dataframe): dataframe for training
        k (int): number of clusters
        additional_info (bool): displays aditional info such as cluster/iteration data
    """
    categorical_cols = ['Country'] # Catergorical columns
    label_encoding_cols = ['StockCode'] # Big columns to encode via label-encoding
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True) # one-hot-encoding

    label_encoder = LabelEncoder() # label-encoding
    df['StockCode_encoded'] = label_encoder.fit_transform(df[label_encoding_cols])
    # While one-hot-encoding provides better results in general, we cant use it in rows with multiples values due to size issues. Therefore we use label-encoding
    
    df = df.drop(columns=["InvoiceNo","StockCode","Description","InvoiceDate","CustomerID"]) # Get rid of all the non-relevant columns

    if (additional_info):
        optimise_k_means(df,10)

    kmeans = KMeans(n_clusters=k, random_state=42) # Train K-means
    df['Cluster'] = kmeans.fit_predict(df)

    scatter_plot(df)

def kmeans_cluster_by_rfm(df: pd.DataFrame, k: int) -> None:

    # Casting and converting datatypes
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    reference_date = df["InvoiceDate"].max()
    df.dropna(subset=["CustomerID"], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(int)

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,  # Recency (days since last purchase)
        "InvoiceNo": "nunique",  # Frequency
        "TotalPrice": "sum"  # Monetary
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"] # Rename columns

    # Normalize the RFM values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # Display first few rows
    pd.DataFrame(rfm_scaled, columns=["Recency", "Frequency", "Monetary"]).head()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Train K-means
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    rfm = rfm.reset_index()

    scatter_plot(rfm)

def scatter_plot(df: pd.DataFrame) -> None:
    """
    Method used to display Kmeans clusters in scatter graphic

    Parameters
    ----------
        df (pd.dataframe): dataframe with trained data
    """
    pca = PCA(n_components=2)# Reduce to 2D
    pca_df = pca.fit_transform(df)

    plt.scatter(pca_df[:, 0], pca_df[:, 1], c=df['Cluster'])
    plt.title('K-means Cluster')
    plt.show()

def print_menu() -> None:
    """
    Shows the main menu
    """
    print("1. Train K-means by encoding")
    print("2. Train K-means by RFM")
    print("3. Exit")

input_file = './data/clean_data.csv'
df = pd.read_csv(input_file, encoding='iso-8859-1')
df.dropna(inplace=True)

while(True):
    print_menu()
    match int(input("User Input: ")):
        case 1:
            k=int(input("Specify of clusters: "))
            boolean = input("Show aditional info? (Y)Yes (N)No: ") == "Y"
            print("Now training Kmeans encoding...")
            kmeans_cluster_by_encoding(df,k,boolean)
        case 2:
            k=int(input("Specify of clusters: "))
            print("Now training Kmeans RFM...")
            kmeans_cluster_by_rfm(df,k)
        case 3:
            print("Exitting...")
            break
        case _:
            print("Unkown input \n")