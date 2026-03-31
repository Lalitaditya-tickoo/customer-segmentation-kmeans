import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Create folders if not exist
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

print("First 5 rows:\n", df.head())


# Select features
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Elbow method
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("plots/elbow_method.png")
plt.close()


# Train final model (k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters


# Save cluster plot
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=clusters)
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.savefig("plots/clusters.png")
plt.close()


# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.title("PCA Clusters")
plt.savefig("plots/pca_clusters.png")
plt.close()


# Save models
joblib.dump(kmeans, "models/kmeans_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("\nTraining complete ✅")