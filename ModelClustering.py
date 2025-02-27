import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("data/clean_data.csv",encoding="latin-1",low_memory=False)  #iso-8859-1

df = df.drop(columns=['InvoiceNo','Description','InvoiceDate','CustomerID','Country'])

df_numeric = df.select_dtypes(include=[np.number])
#Normalizacion

scaler = StandardScaler()
df_normalized = scaler.fit_transform(df_numeric)

#Aplicacion de PCA

# 2. Aplicar PCA, reduciendo a 2 componentes por ejemplo
pca = PCA(n_components=2)
data_pca = pca.fit_transform(df_normalized)

# 3. Ver la varianza explicada por cada componente
print("Varianza explicada:", pca.explained_variance_ratio_)


n_neighbors = 5
neighbors = NearestNeighbors(n_neighbors=n_neighbors)
neighbors_fit = neighbors.fit(data_pca)
distances, indices = neighbors_fit.kneighbors(data_pca)

# Ordenamos las distancias del k-ésimo vecino (por ejemplo, el 5º vecino)
distances_k = np.sort(distances[:, n_neighbors - 1])
plt.plot(distances_k)
plt.xlabel("Puntos ordenados")
plt.ylabel(f"Distancia al {n_neighbors}º vecino")
plt.title("K-distance Plot para determinar eps")
plt.show()

batch_size = 40000  # Ajusta según la memoria disponible
n_batches = len(data_pca) // batch_size

all_clusters = np.full(len(data_pca), -1)  # Inicializamos con -1 (ruido)

for i in range(n_batches + 1):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(data_pca))

    if start >= end:
        break  # Evitar procesar un batch vacío

    print(f"Procesando batch {i + 1}/{n_batches + 1}...")

    batch_data = data_pca[start:end]

# Ejemplo: eps=0.5 y min_samples=5, ajusta según tu k-distance plot
    dbscan = DBSCAN(eps=0.3, min_samples=5, algorithm='kd_tree', n_jobs=1)
    batch_clusters = dbscan.fit_predict(batch_data)

    all_clusters[start:end] = batch_clusters  # Guardamos los clusters en el dataset original

print("Clustering completado en batches.")

# Clusters será un array con las etiquetas asignadas a cada punto.
# Nota: La etiqueta -1 se asigna a puntos considerados ruido.
print("Clusters únicos:", set(all_clusters))

plt.figure(figsize=(8,6))
plt.scatter(data_pca[:,0], data_pca[:,1], c=all_clusters, cmap='viridis', s=50)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clustering con DBSCAN")
plt.colorbar(label="Etiqueta de Cluster")
plt.show()
