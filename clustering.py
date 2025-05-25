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


def load_and_prepare_data():
    """Load and prepare data for clustering analysis"""
    try:
        transaction_data = load_transaction_data('data/transactions.parquet')
        if transaction_data is not None:
            print(f"Loaded {len(transaction_data)} transactions")
            df = create_enhanced_behavioral_features(transaction_data)
            print(f"Created features for {len(df)} customers with {len(df.columns)} features")
            return df, transaction_data
        else:
            print("Failed to load transaction data")
            return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def analyze_umap_agglomerative(X_scaled, feature_names, n_clusters=9):
    """
    Comprehensive analysis of UMAP + Agglomerative clustering
    """
    print(f"Starting UMAP + Agglomerative clustering analysis with {n_clusters} clusters...")

    # Apply UMAP for dimensionality reduction
    print("Applying UMAP dimensionality reduction...")
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
    X_umap = umap_reducer.fit_transform(X_scaled)
    print(f"UMAP completed: {X_umap.shape}")

    # Apply Agglomerative Clustering
    print(f"Applying Agglomerative clustering with {n_clusters} clusters...")
    agg_clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = agg_clusterer.fit_predict(X_umap)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_umap, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_umap, cluster_labels)
    print(f"Silhouette score: {silhouette_avg:.3f}")

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
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        if np.sum(cluster_mask) > 0:  # Check if cluster has data points
            cluster_center = X_umap[cluster_mask].mean(axis=0)
            plt.scatter(cluster_center[0], cluster_center[1],
                        marker='x', s=200, c='red', linewidth=3)
            plt.annotate(f'C{i}', cluster_center,
                         xytext=(5, 5), textcoords='offset points',
                         fontweight='bold', color='red')

    # 2. Silhouette Analysis
    ax2 = plt.subplot(2, 3, 2)
    y_lower = 10
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        if len(cluster_silhouette_values) > 0:
            cluster_silhouette_values = np.sort(cluster_silhouette_values)

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
    bars = plt.bar(range(n_clusters), cluster_sizes.values,
                   color=colors[:len(cluster_sizes)], alpha=0.7)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    plt.title('Cluster Size Distribution', fontweight='bold')
    plt.xticks(range(n_clusters))

    # Add value labels on bars
    for bar, size in zip(bars, cluster_sizes.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f'{size}', ha='center', va='bottom', fontweight='bold')

    # 4. UMAP Components Distribution by Cluster
    ax4 = plt.subplot(2, 3, 4)
    for i in range(n_clusters):
        cluster_data = X_umap[cluster_labels == i]
        if len(cluster_data) > 0:
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                        label=f'Cluster {i}', alpha=0.6, s=30, color=colors[i])
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('Clusters in UMAP Space', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 5. Hierarchical Clustering Dendrogram (on UMAP coordinates)
    ax5 = plt.subplot(2, 3, 5)
    # Sample data for dendrogram (use subset if data is large)
    sample_size = min(500, len(X_umap))  # Reduced sample size for performance
    sample_indices = np.random.choice(len(X_umap), sample_size, replace=False)
    X_sample = X_umap[sample_indices]

    try:
        linkage_matrix = linkage(X_sample, method='ward')
        dendrogram(linkage_matrix, truncate_mode='level', p=10, ax=ax5)
        plt.title('Hierarchical Clustering Dendrogram\n(Ward Linkage)', fontweight='bold')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
    except Exception as e:
        plt.text(0.5, 0.5, f'Dendrogram Error:\n{str(e)}',
                 ha='center', va='center', transform=ax5.transAxes)
        plt.title('Dendrogram (Error)', fontweight='bold')

    # 6. Cluster Quality Metrics
    ax6 = plt.subplot(2, 3, 6)

    # Calculate within-cluster sum of squares and silhouette scores for each cluster
    wcss = []
    cluster_silhouettes = []

    for i in range(n_clusters):
        cluster_data = X_umap[cluster_labels == i]
        if len(cluster_data) > 1:
            cluster_center = cluster_data.mean(axis=0)
            wcss.append(np.sum((cluster_data - cluster_center) ** 2))
            cluster_silhouettes.append(sample_silhouette_values[cluster_labels == i].mean())
        else:
            wcss.append(0)
            cluster_silhouettes.append(0)

    # Normalize WCSS for better visualization
    max_wcss = max(wcss) if max(wcss) > 0 else 1
    wcss_normalized = [w / max_wcss for w in wcss]

    # Create dual-axis plot
    ax6_twin = ax6.twinx()

    x_pos = np.arange(n_clusters)
    bars1 = ax6.bar(x_pos - 0.2, wcss_normalized, 0.4, label='Within-Cluster SS (normalized)',
                    color='lightblue', alpha=0.7)
    bars2 = ax6_twin.bar(x_pos + 0.2, cluster_silhouettes, 0.4,
                         label='Avg Silhouette', color='orange', alpha=0.7)

    ax6.set_xlabel('Cluster')
    ax6.set_ylabel('Normalized Within-Cluster SS', color='blue')
    ax6_twin.set_ylabel('Average Silhouette Score', color='orange')
    ax6.set_title('Cluster Quality Metrics', fontweight='bold')
    ax6.set_xticks(range(n_clusters))

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
    print(f"Number of Clusters: {n_clusters}")
    print(f"Total Customers: {len(cluster_labels)}")
    print("\nCluster Distribution:")

    for i in range(n_clusters):
        cluster_size = np.sum(cluster_labels == i)
        if cluster_size > 0:
            cluster_silhouette = sample_silhouette_values[cluster_labels == i].mean()
            percentage = (cluster_size / len(cluster_labels)) * 100
            print(f"  Cluster {i}: {cluster_size:4d} customers ({percentage:5.1f}%) - "
                  f"Silhouette: {cluster_silhouette:.3f}")
        else:
            print(f"  Cluster {i}: {cluster_size:4d} customers (0.0%) - Empty cluster")

    return X_umap, cluster_labels, silhouette_avg


def create_business_segment_profiles(df, cluster_labels, top_features=10):
    """
    Create business-interpretable segment profiles using original features
    """
    print(f"\nCreating business segment profiles...")

    # Select numerical features only
    numerical_features = df.select_dtypes(include=[np.number])

    # Helper function to handle both scalars and series/arrays
    def is_value_present(val):
        if hasattr(val, 'any'):  # Check if it's a Series, array, or list-like
            return pd.notna(val).any()
        return pd.notna(val)  # It's a single scalar value

    cluster_profiles = []
    n_clusters = len(np.unique(cluster_labels))

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = numerical_features[cluster_mask]

        if len(cluster_data) == 0:
            continue

        profile = {
            'Cluster': cluster_id,
            'Size': np.sum(cluster_mask),
            'Percentage': (np.sum(cluster_mask) / len(cluster_labels)) * 100
        }

        # Calculate feature means for this cluster
        feature_means = cluster_data.mean()

        # Add top distinguishing features
        overall_means = numerical_features.mean()
        feature_ratios = feature_means / (overall_means + 1e-10)  # Avoid division by zero

        # Get top features that distinguish this cluster
        top_distinguishing = feature_ratios.abs().nlargest(top_features)

        for feature in top_distinguishing.index:
            profile[f'{feature}'] = feature_means[feature]
            profile[f'{feature}_ratio'] = feature_ratios[feature]

        cluster_profiles.append(profile)

    profiles_df = pd.DataFrame(cluster_profiles)

    # Display profiles
    print("\nCluster Profiles Summary:")
    for _, row in profiles_df.iterrows():
        print(f"\nCluster {int(row['Cluster'])}: {int(row['Size'])} customers ({row['Percentage']:.1f}%)")

        # --- MODIFIED LINE IS HERE ---
        # Use the helper function to get the feature columns
        feature_cols = [col for col in row.index if col.endswith('_ratio') and is_value_present(row[col])]

        if feature_cols:
            ratios = [(col.replace('_ratio', ''), row[col]) for col in feature_cols]
            # Handle the case where a ratio is a Series by taking its mean
            ratios_cleaned = []
            for feature, ratio_val in ratios:
                if hasattr(ratio_val, 'mean'):
                    ratios_cleaned.append((feature, ratio_val.mean()))
                else:
                    ratios_cleaned.append((feature, ratio_val))

            ratios_cleaned.sort(key=lambda x: abs(x[1]), reverse=True)

            print("  Top distinguishing features:")
            for feature, ratio in ratios_cleaned[:5]:
                direction = "higher" if ratio > 1 else "lower"
                print(f"    {feature}: {ratio:.2f}x ({direction} than average)")

    return profiles_df

def find_optimal_clusters(X_scaled, max_clusters=10):
    """Find optimal number of clusters using silhouette analysis"""
    print("Finding optimal number of clusters...")

    # Apply UMAP first
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
    X_umap = umap_reducer.fit_transform(X_scaled)

    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in cluster_range:
        try:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = clusterer.fit_predict(X_umap)
            score = silhouette_score(X_umap, labels)
            silhouette_scores.append(score)
            print(f"  {n_clusters} clusters: Silhouette score = {score:.3f}")
        except Exception as e:
            print(f"  {n_clusters} clusters: Error - {e}")
            silhouette_scores.append(0)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal Clusters')
    plt.grid(True, alpha=0.3)

    # Highlight best score
    best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    plt.axvline(x=best_n_clusters, color='red', linestyle='--', alpha=0.7)
    plt.text(best_n_clusters, best_score + 0.01, f'Best: {best_n_clusters} clusters',
             ha='center', color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"\nOptimal number of clusters: {best_n_clusters} (Silhouette score: {best_score:.3f})")
    return best_n_clusters, X_umap


def main():
    """Main execution function"""
    print("Starting clustering analysis...")

    # Load and prepare data
    df, transaction_data = load_and_prepare_data()
    if df is None:
        return

    # Prepare features for clustering
    print("Preparing features for clustering...")
    X = df.select_dtypes(include=[np.number])

    if X.empty:
        print("No numerical features found!")
        return

    print(f"Using {len(X.columns)} numerical features for {len(X)} customers")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = X.columns.tolist()

    # Set pandas display options for better output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Find optimal number of clusters
    optimal_clusters, X_umap = find_optimal_clusters(X_scaled, max_clusters=8)

    # Perform detailed analysis with optimal clusters (or use 9 clusters as specified)
    X_umap, cluster_labels, silhouette_score = analyze_umap_agglomerative(
        X_scaled, feature_names, n_clusters=9  # Using 9 clusters as specified
    )

    # Create business profiles
    profiles = create_business_segment_profiles(df, cluster_labels)

    # Save results
    try:
        # Create results dataframe
        results_df = df.copy()
        results_df['cluster'] = cluster_labels
        results_df['umap_1'] = X_umap[:, 0]
        results_df['umap_2'] = X_umap[:, 1]

        print(f"\nSaving results...")
        results_df.to_csv('clustering_results.csv', index=True)
        profiles.to_csv('cluster_profiles.csv', index=False)

        print("Results saved to 'clustering_results.csv' and 'cluster_profiles.csv'")

    except Exception as e:
        print(f"Error saving results: {e}")

    print("\nClustering analysis completed!")
    return results_df, profiles


if __name__ == "__main__":
    # Execute main analysis
    results, profiles = main()