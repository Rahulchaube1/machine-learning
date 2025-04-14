# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Loading the dataset
data = pd.read_csv('Customer_Segmentation.csv')

# Data Preprocessing
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])  # Encoding Gender as 0 or 1

# Scaling the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_clusters = kmeans.fit_predict(scaled_data)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(scaled_data)

# Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=5)
agg_clusters = agg_clust.fit_predict(scaled_data)

# Adding the clusters to the original dataset
data['KMeans_Cluster'] = kmeans_clusters
data['DBSCAN_Cluster'] = dbscan_clusters
data['Agglomerative_Cluster'] = agg_clusters

# Visualizing the K-Means clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='Age', y='Annual Income (k$)', hue='KMeans_Cluster', palette='viridis')
plt.title('K-Means Clustering Results')
plt.show()

# Evaluating the models using Silhouette Score
kmeans_silhouette = silhouette_score(scaled_data, kmeans_clusters)
dbscan_silhouette = silhouette_score(scaled_data, dbscan_clusters) if len(set(dbscan_clusters)) > 1 else -1
agg_silhouette = silhouette_score(scaled_data, agg_clusters)

print(f"K-Means Silhouette Score: {kmeans_silhouette}")
print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
print(f"Agglomerative Silhouette Score: {agg_silhouette}")
