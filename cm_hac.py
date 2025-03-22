import pandas as pd 
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import inconsistent
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import numpy as np
from scipy.stats import zscore

df = pd.read_csv("data/clean_data.csv")
df

# Take a random sample of the data, for example, 10%
df_sample = df.sample(frac=0.1, random_state=42)

df_sample['total_spent'] = df['Quantity'] * df['UnitPrice']

# Group by customer (CustomerID)
df_customers = df_sample.groupby('CustomerID').agg({
    'total_spent': 'sum',  # Total spent by customer
    'InvoiceNo': 'nunique',  # Number of unique purchases
    'Quantity': 'sum',  # Total number of products purchased
}).reset_index()

# Standardize the data before performing clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_customers[['total_spent', 'InvoiceNo', 'Quantity']])

# Perform the linkage for hierarchical clustering
Z = linkage(df_scaled, method='ward', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, color_threshold=0.7)
plt.show()

# Truncate the dendrogram

# Plot the truncated dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, color_threshold=0.7, truncate_mode="lastp", p=10, show_leaf_counts=True, show_contracted=True)
plt.show()

# Custom dendrogram

def dendrogram_tune(*args, **kwargs):
    max_d = kwargs.pop("max_d", None)
    if max_d and "color_threshold" not in kwargs:
        kwargs["color_threshold"] = max_d
    annotate_above = kwargs.pop("annotate_above", 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get("no_plot", False):
        plt.title("Hierarchical Clustering with Truncated Dendrogram")
        plt.xlabel("Dataset Index (or cluster size)")
        plt.ylabel("Distance")
        for i, d, c in zip(ddata["icoord"], ddata["dcoord"], ddata["color_list"]):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, "o", c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords="offset points", va="top", ha="center")
    
    if max_d:
        plt.axhline(y=max_d, c="k")
    
    return ddata

dendrogram_tune(Z, truncate_mode="lastp", p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True, annotate_above=10, max_d=46.7)
plt.show()

# Calculate the inconsistency score
depth = 3
incons = inconsistent(Z, depth)
incons[-10:]

# Elbow method - Adjusted
last = Z[-10:, 2]  # Last 10 distances
last_rev = last[::-1]  # Reverse to view in descending order

# Visualize the elbow method
idx = np.arange(1, len(last) + 1)
plt.figure(figsize=(8, 6))
plt.plot(idx, last_rev, marker='o', linestyle='-', color='b')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Distance")
plt.show()

# Visualize the acceleration
acc = np.diff(last, 2)
acc_rev = acc[::-1]
plt.figure(figsize=(8, 6))
plt.plot(idx[:-2] + 1, acc_rev, marker='o', linestyle='-', color='g')
plt.title("Acceleration for Determining the Number of Clusters")
plt.xlabel("Number of clusters")
plt.ylabel("Acceleration")
plt.show()

k = acc_rev.argmax() + 2  # Optimal number of clusters
print(f"The optimal number of clusters is {k}")

# Hierarchical clustering with fcluster
k = 2
clusters = fcluster(Z, k, criterion="maxclust")
clusters

fcluster(Z, 20, depth=10)  # Dimensionality reduction to 2 dimensions
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_customers)

# Cluster plot with PCA
plt.figure(figsize=(10, 8))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters, cmap="prism")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Cluster plot with PCA
plt.figure(figsize=(10, 8))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=clusters, cmap='prism', s=50)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Cluster Visualization (PCA)")
plt.show()

# Plot Quantity and total_spent
plt.figure(figsize=(10, 8))
plt.scatter(df_customers["Quantity"], df_customers["total_spent"], c=clusters, cmap="prism")
plt.xlabel("Quantity")
plt.ylabel("total_spent")
plt.show()

# Plot total_spent and InvoiceNo
plt.figure(figsize=(10, 8))
plt.scatter(df_customers["total_spent"], df_customers["InvoiceNo"], c=clusters, cmap="prism")
plt.xlabel("Total spent")
plt.ylabel("Number of purchases")
plt.show()

# Plot Quantity and InvoiceNo
plt.figure(figsize=(10, 8))
plt.scatter(df_customers["Quantity"], df_customers["InvoiceNo"], c=clusters, cmap="prism")
plt.xlabel("Number of products purchased")
plt.ylabel("Number of purchases")
plt.show()


# Calculate the Z-score for numeric columns
z_scores = np.abs(zscore(df_customers[['total_spent', 'Quantity', 'InvoiceNo']]))

# Check how many points have a Z-score greater than 3
outliers = (z_scores > 3).all(axis=1)  # If the Z-score is greater than 3 in all columns
outliers_count = np.sum(outliers)
print(f"Number of outliers: {outliers_count}")

# Filter outliers if desired
df_customers_no_outliers = df_customers[~outliers]

# Filter outliers
df_customers_no_outliers = df_customers[~outliers]

# Perform linkage again with the data without outliers
df_scaled_no_outliers = scaler.fit_transform(df_customers_no_outliers[['total_spent', 'InvoiceNo', 'Quantity']])
Z_no_outliers = linkage(df_scaled_no_outliers, method='ward', metric='euclidean')

# Dendrogram of the data without outliers
plt.figure(figsize=(10, 7))
dendrogram(Z_no_outliers, color_threshold=0.7)
plt.show()

# Perform clustering with the new linkage
clusters_no_outliers = fcluster(Z_no_outliers, k, criterion="maxclust")

# Visualize the combinations of variables
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].scatter(df_customers['total_spent'], df_customers['Quantity'], c=clusters, cmap='prism')
ax[0].set_xlabel("Total Spent")
ax[0].set_ylabel("Number of Products Purchased")

ax[1].scatter(df_customers['total_spent'], df_customers['InvoiceNo'], c=clusters, cmap='prism')
ax[1].set_xlabel("Total Spent")
ax[1].set_ylabel("Number of Purchases")

ax[2].scatter(df_customers['Quantity'], df_customers['InvoiceNo'], c=clusters, cmap='prism')
ax[2].set_xlabel("Number of Products Purchased")
ax[2].set_ylabel("Number of Purchases")

plt.tight_layout()
plt.show()

# Show the efficiency of our hierarchical clustering, the closer to 1, the more accurate our cluster is
# Calculate the Silhouette Score
silhouette_avg = silhouette_score(df_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")

sample_silhouette_values = silhouette_samples(df_scaled, clusters)

plt.figure(figsize=(10, 7))
plt.hist(sample_silhouette_values, bins=10, color='skyblue', edgecolor='black')
plt.title("Silhouette Score Distribution")
plt.xlabel("Silhouette Score")
plt.ylabel("Number of Samples")
plt.show()