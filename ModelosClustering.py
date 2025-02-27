import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./data/clean_data.csv', encoding='iso-8859-1')
df.dropna(inplace=True)

categorical_cols = ['Country']
encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True) # one-hot-encoding o algo asi

label_encoder = LabelEncoder() # label encoding, que no level
encoded_df['StockCode_encoded'] = label_encoder.fit_transform(encoded_df['StockCode'])

encoded_df = encoded_df.drop(columns=["InvoiceNo","StockCode","Description","InvoiceDate","CustomerID"])
#print(encoded_df.head())
#encoded_df.to_csv("C:/Users/Fernando/Desktop/datasets/data.csv")

#encoded_df['StockCode'] = pd.to_numeric(encoded_df['StockCode'], errors='coerce')
#df['StockCode'] = pd.to_numeric(df['StockCode'], errors='coerce')
#features = df[['Quantity', 'UnitPrice', 'CustomerID']]
features = encoded_df

#scaler = StandardScaler()
#features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
encoded_df['Cluster'] = kmeans.fit_predict(features)

print(encoded_df.head())


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
features_pca = pca.fit_transform(features)

# Visualizar los clusters
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=encoded_df['Cluster'], cmap='viridis')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Clusters de K-Means')
plt.show()
