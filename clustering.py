# UMAP + Agglomerative Clustering Analysis and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
from data_loader import load_transaction_data
from feature_engineering import create_enhanced_behavioral_features

warnings.filterwarnings('ignore')
transaction_data = load_transaction_data('data/transactions.parquet')
if transaction_data is not None:
    df = create_enhanced_behavioral_features(transaction_data)
else:
    print("Failed to load transaction data")
    exit(1)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

X = df.select_dtypes(include=np.number)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

feature_names = df.select_dtypes(include=np.number).columns.tolist()


# Assuming you have your processed data in 'df' and numerical features in 'X'
# This code should be run after your main.py data preparation

def analyze_umap_agglomerative(X, df=None):
    """
    Comprehensive analysis of UMAP + Agglomerative clustering
    """

    # Apply UMAP
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
    X_umap = umap_reducer.fit_transform(X_scaled)

    # Apply Agglomerative Clustering
    agg_clusterer = AgglomerativeClustering(n_clusters=9)
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
    for i in range(10):
        cluster_center = X_umap[cluster_labels == i].mean(axis=0)
        plt.scatter(cluster_center[0], cluster_center[1],
                    marker='x', s=200, c='red', linewidth=3)
        plt.annotate(f'Cluster {i}', cluster_center,
                     xytext=(5, 5), textcoords='offset points',
                     fontweight='bold', color='red')

    # 2. Silhouette Analysis
    ax2 = plt.subplot(2, 3, 2)
    y_lower = 10
    colors = plt.cm.viridis(np.linspace(0, 1, 5))

    for i in range(6):
        cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        cluster_silhouette_values.sort()

        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                          facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.axvline(x=silhouette_avg, color="red", linestyle="--",
                label=f'Average Score: {silhouette_avg:.3f}')
    plt.xlabel('Silhouette Coefficient Values')
    plt.ylabel('Cluster Label')
    plt.title('Silhouette Analysis', fontweight='bold')
    plt.legend()

    # 3. Cluster Size Distribution
    ax3 = plt.subplot(2, 3, 3)
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    bars = plt.bar(range(10), cluster_sizes.values, color=colors, alpha=0.7)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    plt.title('Cluster Size Distribution', fontweight='bold')
    plt.xticks(range(10))

    # Add value labels on bars
    for bar, size in zip(bars, cluster_sizes.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f'{size}', ha='center', va='bottom', fontweight='bold')

    # 4. UMAP Components Distribution
    ax4 = plt.subplot(2, 3, 4)
    for i in range(10):
        cluster_data = X_umap[cluster_labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    label=f'Cluster {i}', alpha=0.6, s=30)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('Clusters in UMAP Space', fontweight='bold')
    plt.legend()

    # 5. Hierarchical Clustering Dendrogram (on UMAP coordinates)
    ax5 = plt.subplot(2, 3, 5)
    # Sample data for dendrogram (use subset if data is large)
    sample_size = min(1000, len(X_umap))
    sample_indices = np.random.choice(len(X_umap), sample_size, replace=False)
    X_sample = X_umap[sample_indices]

    linkage_matrix = linkage(X_sample, method='ward')
    dendrogram(linkage_matrix, truncate_mode='level', p=10, ax=ax5)
    plt.title('Hierarchical Clustering Dendrogram\n(Ward Linkage)', fontweight='bold')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')

    # 6. Cluster Quality Metrics
    ax6 = plt.subplot(2, 3, 6)

    # Calculate within-cluster sum of squares for each cluster
    wcss = []
    cluster_silhouettes = []

    for i in range(10):
        cluster_data = X_umap[cluster_labels == i]
        if len(cluster_data) > 1:
            cluster_center = cluster_data.mean(axis=0)
            wcss.append(np.sum((cluster_data - cluster_center) ** 2))
            cluster_silhouettes.append(sample_silhouette_values[cluster_labels == i].mean())
        else:
            wcss.append(0)
            cluster_silhouettes.append(0)

    # Create dual-axis plot
    ax6_twin = ax6.twinx()

    bars1 = ax6.bar(np.arange(5) - 0.2, wcss, 0.4, label='Within-Cluster SS',
                    color='lightblue', alpha=0.7)
    bars2 = ax6_twin.bar(np.arange(5) + 0.2, cluster_silhouettes, 0.4,
                         label='Avg Silhouette', color='orange', alpha=0.7)

    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('Within-Cluster Sum of Squares', color='blue')
    ax6_twin.set_ylabel('Average Silhouette Score', color='orange')
    ax6.set_title('Cluster Quality Metrics', fontweight='bold')
    ax6.set_xticks(range(5))

    # Add legends
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Print cluster analysis
    print("=" * 60)
    print("UMAP + AGGLOMERATIVE CLUSTERING ANALYSIS")
    print("=" * 60)
    print(f"Overall Silhouette Score: {silhouette_avg:.3f}")
    print(f"Number of Clusters: 5")
    print(f"Total Customers: {len(cluster_labels)}")
    print("\nCluster Distribution:")
    for i in range(5):
        cluster_size = np.sum(cluster_labels == i)
        cluster_silhouette = sample_silhouette_values[cluster_labels == i].mean()
        percentage = (cluster_size / len(cluster_labels)) * 100
        print(f"  Cluster {i}: {cluster_size:4d} customers ({percentage:5.1f}%) - "
              f"Silhouette: {cluster_silhouette:.3f}")

    return X_umap, cluster_labels, silhouette_avg


# Example usage (uncomment and run after loading your data):
X_umap, labels, score = analyze_umap_agglomerative(X_scaled, df)


def create_business_segment_profiles(X, cluster_labels, feature_names=None):
    """
    Create business-interpretable segment profiles
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    # Calculate cluster centers in original feature space
    cluster_profiles = []

    for cluster_id in range(10):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = X[cluster_mask]

        profile = {
            'Cluster': cluster_id,
            'Size': np.sum(cluster_mask),
            'Percentage': (np.sum(cluster_mask) / len(cluster_labels)) * 100
        }

        # Add feature statistics
        for i, feature in enumerate(feature_names):
            profile[f'{feature}_mean'] = cluster_data[feature].mean()
            profile[f'{feature}_std'] = cluster_data[feature].std()

        cluster_profiles.append(profile)


    return pd.DataFrame(cluster_profiles)

# Usage example:
profiles = create_business_segment_profiles(X, labels, feature_names)
print(profiles)
