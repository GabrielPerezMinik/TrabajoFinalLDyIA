import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

class CDbscan:
    # =============================================================================
    # Functions for PCA and neighbor analysis (global dataset processing)
    # =============================================================================
    @staticmethod
    def load_data(filepath):
        """
        Loads the CSV file and drops unwanted columns.
        Note: This function is used for global analysis (e.g., PCA) and not for batch processing.

        :param filepath: Path to the CSV file.
        :return: DataFrame with selected columns removed.
        """
        df = pd.read_csv(filepath, encoding="iso-8859-1", low_memory=False)
        # Drop columns not needed for PCA analysis (e.g., CustomerID, InvoiceNo, etc.)
        df = df.drop(columns=['InvoiceNo', 'Description', 'CustomerID', 'InvoiceDate', 'Country'])
        return df

    @staticmethod
    def normalize_data(df):
        """
        Selects numeric columns and scales them using StandardScaler.

        :param df: DataFrame to be normalized.
        :return: Numpy array of normalized data.
        """
        df_numeric = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        df_normalized = scaler.fit_transform(df_numeric)
        return df_normalized

    @staticmethod
    def apply_pca(df_normalized, n_components=2):
        """
        Applies Principal Component Analysis (PCA) to reduce the dimensionality.

        :param df_normalized: Normalized data.
        :param n_components: Number of principal components.
        :return: Transformed data with reduced dimensions.
        """
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(df_normalized)
        print("Explained variance:", pca.explained_variance_ratio_)
        return data_pca

    @staticmethod
    def calculate_neighbors(data_pca, n_neighbors=5):
        """
        Computes the nearest neighbors for the PCA-transformed data.

        :param data_pca: PCA-transformed data.
        :param n_neighbors: Number of neighbors to consider.
        :return: Array of distances from each point to its nth neighbor.
        """
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors_fit = neighbors.fit(data_pca)
        distances, indices = neighbors_fit.kneighbors(data_pca)
        return distances

    @staticmethod
    def plot_distances(distances, n_neighbors=5):
        """
        Plots the sorted distances to the nth neighbor to help determine the eps parameter for DBSCAN.

        :param distances: Array of distances from each point to its nth neighbor.
        :param n_neighbors: The neighbor index used.
        """
        distances_k = np.sort(distances[:, n_neighbors - 1])
        plt.plot(distances_k)
        plt.xlabel("Ordered points")
        plt.ylabel(f"Distance to the {n_neighbors}th neighbor")
        plt.title("K-distance Plot for determining eps")
        plt.show()

    # =============================================================================
    # Common functions for scaling and clustering
    # =============================================================================
    @staticmethod
    def apply_clustering(data, scaler, eps=0.5, min_samples=5):
        """
        Scales the data using the provided scaler and applies DBSCAN clustering.

        :param data: DataFrame with features to be clustered.
        :param scaler: Scaler instance (e.g., StandardScaler, MinMaxScaler).
        :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
        :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        :return: DataFrame with an added 'Cluster' column containing the cluster labels.
        """
        scaled_data = scaler.fit_transform(data)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_data)
        data = data.copy()
        data['Cluster'] = clusters
        return data


    # =============================================================================
    # Generic functions for batch processing
    # =============================================================================
    @staticmethod
    def process_batches(filepath, batch_size, preprocess_func, encoding='iso-8859-1'):
        """
        Reads the CSV file in chunks and applies the specified preprocessing function to each batch.

        :param filepath: Path to the CSV file.
        :param batch_size: Number of rows per batch.
        :param preprocess_func: Preprocessing function to apply on each batch.
        :param encoding: File encoding.
        :return: DataFrame with the processed and concatenated results from all batches.
        """
        reader = pd.read_csv(filepath, chunksize=batch_size, encoding=encoding)
        results = []
        batch_counter = 0

        for chunk in reader:
            batch_counter += 1
            print(f"Processing batch {batch_counter}...")
            processed = preprocess_func(chunk, batch_counter)
            if processed is not None and not processed.empty:
                results.append(processed)
        if results:
            final_df = pd.concat(results, ignore_index=True)
            return final_df
        else:
            return pd.DataFrame()


    # =============================================================================
    # Specific preprocessing functions for each case
    # =============================================================================
    @staticmethod
    def preprocess_batch_type1(chunk, batch_number=None):
        """
        Preprocessing for Type 1:
          - Converts 'InvoiceDate' to datetime and calculates 'DaysSinceFirstPurchase'.
          - Selects columns: ['Quantity', 'UnitPrice', 'CustomerID', 'DaysSinceFirstPurchase'].
          - Drops rows with missing values.
          - Scales using MinMaxScaler and applies DBSCAN.

        :param chunk: DataFrame chunk.
        :param batch_number: Batch number (optional).
        :return: Processed DataFrame with clustering applied.
        """
        # Convert date and calculate new feature
        chunk["InvoiceDate"] = pd.to_datetime(chunk["InvoiceDate"], errors="coerce")
        chunk["DaysSinceFirstPurchase"] = (chunk["InvoiceDate"] - chunk["InvoiceDate"].min()).dt.days

        # Select relevant columns and drop rows with missing values
        cols = ["Quantity", "UnitPrice", "CustomerID", "DaysSinceFirstPurchase"]
        chunk = chunk[cols].dropna()

        if chunk.empty:
            print("Batch is empty after cleaning. Skipping...")
            return None

        # Apply scaling and clustering using MinMaxScaler
        processed = CDbscan.apply_clustering(chunk, scaler=MinMaxScaler(), eps=0.5, min_samples=5)
        return processed

    @staticmethod
    def preprocess_batch_type2(chunk, batch_number):
        """
        Preprocessing for Type 2:
          - Drops rows with missing values in ['Quantity', 'UnitPrice', 'CustomerID'].
          - Selects these columns for clustering.
          - Scales using StandardScaler and applies DBSCAN.
          - Adds a 'BatchNumber' column.
          - Retains other columns (e.g., 'Country') for further analysis.

        :param chunk: DataFrame chunk.
        :param batch_number: Current batch number.
        :return: Processed DataFrame with clustering and batch number information.
        """
        batch_clean = chunk.dropna(subset=['Quantity', 'UnitPrice', 'CustomerID']).copy()
        if batch_clean.empty:
            print("Batch is empty after cleaning. Skipping...")
            return None

        # Select features for clustering
        features = batch_clean[['Quantity', 'UnitPrice', 'CustomerID']]
        # Apply clustering using StandardScaler
        clustered_features = CDbscan.apply_clustering(pd.DataFrame(features, columns=features.columns),
                                                    scaler=StandardScaler(), eps=0.5, min_samples=5)
        # Assign the 'Cluster' column to the original DataFrame
        batch_clean['Cluster'] = clustered_features['Cluster']
        # Add batch number information
        batch_clean['BatchNumber'] = batch_number
        return batch_clean

    # =============================================================================
    # Other functions for saving, plotting, and analysis
    # =============================================================================
    @staticmethod
    def save_results(final_df, output_file="dbscan_batches_result.csv"):
        """
        Saves the final DataFrame to a CSV file.

        :param final_df: DataFrame containing the final clustering results.
        :param output_file: Name of the output CSV file.
        """
        final_df.to_csv(output_file, index=False)
        print(f"Process completed. Results saved in '{output_file}'")

    @staticmethod
    def plot_clusters(final_df):
        """
        Plots a scatter plot of Quantity vs. Unit Price colored by cluster.

        :param final_df: DataFrame containing the clustering results.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=final_df["Quantity"],
            y=final_df["UnitPrice"],
            hue=final_df["Cluster"],
            palette="viridis",
            alpha=0.5
        )
        plt.title("Cluster Visualization (DBSCAN)")
        plt.xlabel("Quantity (Normalized)")
        plt.ylabel("Unit Price (Normalized)")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    @staticmethod
    def analyze_clustering_results(results):
        """
        Analyzes the clustering results by summarizing the number of clusters, noise points,
        and exporting results and summaries to CSV files. Also generates a scatter plot if feasible.

        :param results: DataFrame with clustering results.
        """
        print("Total number of clusters:", len(set(results['Cluster'])) - (1 if -1 in results['Cluster'] else 0))
        print("Number of noise points:", list(results['Cluster']).count(-1))

        # Create a summary of clusters (excluding noise)
        cluster_summary = results[results['Cluster'] != -1].groupby('Cluster').agg({
            'Quantity': ['mean', 'min', 'max', 'count'],
            'UnitPrice': ['mean', 'min', 'max'],
            'CustomerID': 'nunique'
        })
        print("\nCluster Summary:")
        print(cluster_summary)

        # Analyze cluster distribution by country if available
        if 'Country' in results.columns:
            country_cluster_dist = results.groupby(['Country', 'Cluster']).size().unstack(fill_value=0)
            print("\nCluster Distribution by Country:")
            print(country_cluster_dist)
            country_cluster_dist.to_csv('country_cluster_distribution.csv')
        else:
            print("\n'Country' column not found for country-based analysis.")

        # Export results and summary to CSV files
        results.to_csv('clustering_results.csv', index=False)
        cluster_summary.to_csv('cluster_summary.csv')

        # Generate a scatter plot if the dataset is manageable
        if len(results) < 100000000:  # Arbitrary limit to avoid overload
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                results['Quantity'],
                results['UnitPrice'],
                c=results['Cluster'],
                cmap='viridis',
                alpha=0.7
            )
            plt.colorbar(scatter)
            plt.title('Sales Data Clustering')
            plt.xlabel('Quantity')
            plt.ylabel('Unit Price')
            plt.savefig('cluster_visualization.png')
            plt.close()


# =============================================================================
# Main execution block
# =============================================================================
#if __name__ == "__main__":

def run(filepath="data/clean_data.csv"):
    #filepath = "data/clean_data.csv"

    # --- Step 1: Global analysis with PCA and neighbors ---
    cdb = CDbscan()
    df = cdb.load_data(filepath)
    df_normalized = cdb.normalize_data(df)
    data_pca = cdb.apply_pca(df_normalized)
    distances = cdb.calculate_neighbors(data_pca)
    cdb.plot_distances(distances)

    # --- Step 2: Batch processing using Type 1 method ---
    final_df_type1 = cdb.process_batches(filepath, batch_size=50000, preprocess_func=cdb.preprocess_batch_type1,
                                     encoding='iso-8859-1')
    cdb.save_results(final_df_type1, output_file="ClusterData/dbscan_batches_result_type1.csv")
    cdb.plot_clusters(final_df_type1)

    # --- Step 3: Batch processing using Type 2 method ---
    try:
        final_df_type2 = cdb.process_batches(filepath, batch_size=50000, preprocess_func=cdb.preprocess_batch_type2,
                                         encoding='unicode_escape')
        cdb.analyze_clustering_results(final_df_type2)
        print("\nProcess completed. Results have been saved in CSV files.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()

