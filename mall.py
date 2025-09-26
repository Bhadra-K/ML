import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Load dataset
data = pd.read_csv("Mall_Customers.csv")
print(data.head())
# Features for clustering
x=data[['Annual Income (k$)','Spending Score (1-100)']]
#Feature Scaling (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
# Apply KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
# Add cluster labels to the original data
data['Cluster'] = kmeans.labels_
# Visualize the Clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'],
            c=data['Cluster'], cmap='viridis', s=100)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments (K-Means Clustering)")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
