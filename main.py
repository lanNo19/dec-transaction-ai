# main.py
from data_loader import load_transaction_data
from feature_engineering import create_behavioral_features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    transaction_data = load_transaction_data('data/transactions.parquet')
    if transaction_data is not None:
        df = create_behavioral_features(transaction_data)
    else:
        print("Failed to load transaction data")
        exit(1)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    X = df.select_dtypes(include=np.number)

    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Dimensionality reduction methods
    reduction_methods = {
        'None': lambda X: X,
        'PCA': lambda X: PCA(n_components=10).fit_transform(X),
        'UMAP': lambda X: umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
    }

    # Clustering algorithms that support variable k
    param_clustering_algorithms = {
        'GaussianMixture': lambda X, k: GaussianMixture(n_components=k, random_state=42).fit(X).predict(X),
        'Agglomerative': lambda X, k: AgglomerativeClustering(n_clusters=k).fit_predict(X),
        'Spectral': lambda X, k: SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42).fit_predict(X),
        'KMeans': lambda X, k: KMeans(n_clusters=k, random_state=42).fit_predict(X)
    }

    results = []
    cluster_range = range(2, 11)

    for red_name, reducer in reduction_methods.items():
        X_reduced = reducer(X_scaled)

        for cluster_name, clusterer in param_clustering_algorithms.items():
            for k in cluster_range:
                try:
                    labels = clusterer(X_reduced, k)
                    if len(set(labels)) > 1:
                        score = silhouette_score(X_reduced, labels)
                    else:
                        score = -1
                    results.append({
                        'Reduction': red_name,
                        'Clustering': cluster_name,
                        'NumClusters': k,
                        'Silhouette': score
                    })
                except Exception as e:
                    print(f"Error with {cluster_name} at k={k}: {e}")

    # Create DataFrame and plot
    results_df = pd.DataFrame(results)
    print(results_df[(results_df['Reduction']=='UMAP')&(results_df['Clustering']=='Agglomerative')])

    plt.figure(figsize=(12, 6))
    for (red, cluster), group in results_df.groupby(['Reduction', 'Clustering']):
        plt.plot(group['NumClusters'], group['Silhouette'], marker='o', label=f"{red} + {cluster}")

    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Apply UMAP
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
    X_umap = umap_reducer.fit_transform(X_scaled)

    # Apply Agglomerative Clustering
    agg_clusterer = AgglomerativeClustering(n_clusters=6)
    cluster_labels = agg_clusterer.fit_predict(X_umap)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_umap, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_umap, cluster_labels)

    # Create comprehensive plots
    fig = plt.figure(figsize=(20, 15))

    # 1. UMAP Scatter Plot with Clusters
    ax1 = plt.subplot(2, 3, 1)
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels,
                          cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter)
    plt.title(f'UMAP + Agglomerative Clustering\n(Silhouette Score: {silhouette_avg:.3f})',
              fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')

    # Add cluster centers
    for i in range(7):
        cluster_center = X_umap[cluster_labels == i].mean(axis=0)
        plt.scatter(cluster_center[0], cluster_center[1],
                    marker='x', s=200, c='red', linewidth=3)
        plt.annotate(f'Cluster {i}', cluster_center,
                     xytext=(5, 5), textcoords='offset points',
                     fontweight='bold', color='red')
    plt.show()