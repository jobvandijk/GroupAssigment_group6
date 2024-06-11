import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from joblib import dump

# Generate synthetic data
X, _ = make_blobs(n_samples=3000, centers=6, cluster_std=0.60, random_state=0)

# Create and fit the KMeans model
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
plt.plot(range(1,11),inertias,marker='o')
plt.title("Inertia Plot")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(X)
inertias.append(kmeans.inertia_)

# Visualize the clusters
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title("Clusters Visualization")
plt.show()



#saving the k-means model
dump(kmeans, 'kmeans.joblib')

# Example new data points
new_points = np.array([[0, 2], [1, -1], [-1, 2]])

# Use the loaded model to predict clusters for new data points
predicted_clusters = kmeans.predict(new_points)

print(f"Predicted cluster for each new point: {predicted_clusters}")

