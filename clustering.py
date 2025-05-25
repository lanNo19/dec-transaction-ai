# UMAP + Agglomerative Clustering Analysis and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import \
    StandardScaler  # Kept for type hinting if preprocess_features_for_clustering returns it
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
# from scipy.spatial.distance import pdist # Not explicitly used in the provided code, can be removed if not needed elsewhere
import warnings
from data_loader import load_transaction_data
from feature_engineering import (
    create_enhanced_behavioral_features,
    remove_low_variance_features,
    remove_highly_correlated_features,
    preprocess_features_for_clustering
)

warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load and prepare data for clustering analysis"""
    try:
        # Assuming 'data/transactions.parquet' is the correct path as in the original script
        transaction_data = load_transaction_data('data/transactions.parquet')
        if transaction_data is not None:
            print(f"Loaded {len(transaction_data)} transactions")
            # df will contain card_id as index and enhanced behavioral features
            df = create_enhanced_behavioral_features(transaction_data)
            if df is None or df.empty:
                print("Feature creation resulted in an empty or None DataFrame.")
                return None, None
            print(f"Created features for {len(df)} customers with {len(df.columns)} features")
            return df, transaction_data  # df here is the one with card_id index
        else:
            print("Failed to load transaction data")
            return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def analyze_umap_agglomerative(X_processed, feature_names, n_clusters=14):  # Default n_clusters updated
    """
    Comprehensive analysis of UMAP + Agglomerative clustering.
    X_processed should be the data ready for UMAP (e.g., scaled, after dimensionality reduction if any prior to UMAP).
    """
    print(f"Starting UMAP + Agglomerative clustering analysis with {n_clusters} clusters...")

    # Apply UMAP for dimensionality reduction
    print("Applying UMAP dimensionality reduction...")
    # Ensure X_processed is suitable for UMAP (e.g., no NaNs, numeric)
    if pd.DataFrame(X_processed).isnull().any().any():
        print("Warning: Data passed to UMAP contains NaNs. UMAP might fail or produce unexpected results.")
        # Optionally, handle NaNs here, e.g., by imputation if not handled before
        # X_processed = pd.DataFrame(X_processed).fillna(pd.DataFrame(X_processed).mean()).values

    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
    X_umap = umap_reducer.fit_transform(X_processed)
    print(f"UMAP completed: {X_umap.shape}")

    # Apply Agglomerative Clustering
    print(f"Applying Agglomerative clustering with {n_clusters} clusters...")
    agg_clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels = agg_clusterer.fit_predict(X_umap)

    # Calculate silhouette score
    # Ensure there's more than 1 unique label and more than 1 sample for silhouette score
    if len(np.unique(cluster_labels)) > 1 and len(X_umap) > 1:
        silhouette_avg = silhouette_score(X_umap, cluster_labels)
        sample_silhouette_values = silhouette_samples(X_umap, cluster_labels)
        print(f"Silhouette score: {silhouette_avg:.3f}")
    else:
        silhouette_avg = -1  # Or some other indicator of invalid score
        sample_silhouette_values = np.zeros(len(X_umap))
        print("Silhouette score cannot be computed (less than 2 clusters or insufficient samples).")

    # Create comprehensive plots
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle(f"UMAP + Agglomerative Clustering (Target: {n_clusters} Clusters)", fontsize=16, fontweight='bold')

    # 1. UMAP Scatter Plot with Clusters
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels,
                          cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, ax=ax1)
    ax1.set_title(f'UMAP Projection with Clusters\n(Silhouette: {silhouette_avg:.3f})',
                  fontsize=14)
    ax1.set_xlabel('UMAP Component 1')
    ax1.set_ylabel('UMAP Component 2')

    # Add cluster centers
    unique_labels = np.unique(cluster_labels)
    for i in unique_labels:
        cluster_mask = cluster_labels == i
        if np.sum(cluster_mask) > 0:
            cluster_center = X_umap[cluster_mask].mean(axis=0)
            ax1.scatter(cluster_center[0], cluster_center[1],
                        marker='x', s=200, c='red', linewidth=3)
            ax1.annotate(f'C{i}', cluster_center,
                         xytext=(5, 5), textcoords='offset points',
                         fontweight='bold', color='red')

    # 2. Silhouette Analysis
    ax2 = plt.subplot(2, 3, 2)
    y_lower = 10
    # Use a consistent color map based on the actual number of unique clusters found
    # This handles cases where AgglomerativeClustering might produce fewer clusters than requested
    # if data is very sparse or n_samples < n_clusters.
    actual_n_clusters = len(unique_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, actual_n_clusters))

    # Map original cluster labels to a 0-indexed range for color consistency if labels are not contiguous
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    if silhouette_avg != -1 and actual_n_clusters > 1:  # Only plot if score is valid
        for k, cluster_k_label in enumerate(unique_labels):  # Iterate over actual unique labels
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == cluster_k_label]
            # cluster_silhouette_values might be empty if a label in unique_labels somehow doesn't map, though unlikely
            if len(cluster_silhouette_values) > 0:
                cluster_silhouette_values = np.sort(cluster_silhouette_values)
                size_cluster_i = cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color_idx = label_map[cluster_k_label]  # get mapped index for color
                ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                                  facecolor=colors[color_idx], edgecolor=colors[color_idx], alpha=0.7)
                ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_k_label))
                y_lower = y_upper + 10  # Add a small gap

        ax2.axvline(x=silhouette_avg, color="red", linestyle="--",
                    label=f'Average Score: {silhouette_avg:.3f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Silhouette analysis not applicable\n(e.g. < 2 clusters found)',
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

    ax2.set_xlabel('Silhouette Coefficient Values')
    ax2.set_ylabel('Cluster Label')
    ax2.set_title('Silhouette Analysis', fontweight='bold')

    # 3. Cluster Size Distribution
    ax3 = plt.subplot(2, 3, 3)
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    # Ensure colors match the number of bars if actual_n_clusters is less than n_clusters
    bar_colors = [colors[label_map[label]] for label in cluster_sizes.index]
    bars = ax3.bar(cluster_sizes.index.astype(str), cluster_sizes.values,
                   color=bar_colors, alpha=0.7)  # Use actual labels for x-ticks
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Number of Customers')
    ax3.set_title('Cluster Size Distribution', fontweight='bold')
    # ax3.set_xticks(cluster_sizes.index) # Set x-ticks to actual cluster labels
    # ax3.set_xticklabels(cluster_sizes.index.astype(str))

    for bar_obj, size in zip(bars, cluster_sizes.values):  # bar is now matplotlib.patches.Rectangle
        ax3.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + max(1, 0.01 * bar_obj.get_height()),
                 # Dynamic offset
                 f'{size}', ha='center', va='bottom', fontweight='bold')

    # 4. UMAP Components Distribution by Cluster (similar to plot 1 but for individual plotting)
    ax4 = plt.subplot(2, 3, 4)
    for k, cluster_k_label in enumerate(unique_labels):
        cluster_data = X_umap[cluster_labels == cluster_k_label]
        if len(cluster_data) > 0:
            color_idx = label_map[cluster_k_label]
            ax4.scatter(cluster_data[:, 0], cluster_data[:, 1],
                        label=f'Cluster {cluster_k_label}', alpha=0.6, s=30, color=colors[color_idx])
    ax4.set_xlabel('UMAP Component 1')
    ax4.set_ylabel('UMAP Component 2')
    ax4.set_title('Clusters in UMAP Space (Detail)', fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clusters")

    # 5. Hierarchical Clustering Dendrogram (on UMAP coordinates)
    ax5 = plt.subplot(2, 3, 5)
    sample_size = min(500, len(X_umap))
    if sample_size > 1:  # Dendrogram needs at least 2 samples
        sample_indices = np.random.choice(len(X_umap), sample_size, replace=False)
        X_sample = X_umap[sample_indices]
        try:
            linkage_matrix = linkage(X_sample, method='ward')
            dendrogram(linkage_matrix, truncate_mode='level', p=10, ax=ax5)
            ax5.set_title('Hierarchical Clustering Dendrogram\n(Ward Linkage, UMAP Sample)', fontweight='bold')
            ax5.set_xlabel('Sample Index or (Cluster Size)')
            ax5.set_ylabel('Distance')
        except Exception as e:
            ax5.text(0.5, 0.5, f'Dendrogram Error:\n{str(e)[:100]}...',  # Truncate long error messages
                     ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Dendrogram (Error)', fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'Not enough data for Dendrogram',
                 ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Dendrogram (Skipped)', fontweight='bold')

    # 6. Cluster Quality Metrics
    ax6 = plt.subplot(2, 3, 6)
    wcss = []  # Within-cluster sum of squares on UMAP data
    cluster_silhouettes = []

    if actual_n_clusters > 0:
        for cluster_k_label in unique_labels:
            cluster_data_umap = X_umap[cluster_labels == cluster_k_label]
            if len(cluster_data_umap) > 1 and silhouette_avg != -1:  # Need >1 for WCSS and silhouette
                cluster_center_umap = cluster_data_umap.mean(axis=0)
                wcss.append(np.sum((cluster_data_umap - cluster_center_umap) ** 2))
                cluster_silhouettes.append(sample_silhouette_values[cluster_labels == cluster_k_label].mean())
            elif len(cluster_data_umap) == 1 and silhouette_avg != -1:  # Single point cluster
                wcss.append(0)
                cluster_silhouettes.append(
                    sample_silhouette_values[cluster_labels == cluster_k_label].mean())  # Silhouette can be computed
            else:  # Empty or invalid for metrics
                wcss.append(0)
                cluster_silhouettes.append(0 if silhouette_avg != -1 else -1)  # Use -1 if overall silhouette is invalid

        # Normalize WCSS for better visualization if wcss is not empty
        if wcss:
            max_wcss = max(wcss) if max(wcss) > 0 else 1
            wcss_normalized = [w / max_wcss for w in wcss]
        else:  # Handle empty wcss list (e.g. if no clusters have > 1 point)
            wcss_normalized = [0] * actual_n_clusters

        ax6_twin = ax6.twinx()
        x_pos = np.arange(actual_n_clusters)

        bar_colors_wcss = [colors[label_map[label]] for label in unique_labels]
        bar_colors_sil = [colors[label_map[label]] for label in unique_labels]  # Potentially different if needed

        bars1 = ax6.bar(x_pos - 0.2, wcss_normalized[:actual_n_clusters], 0.4, label='Within-Cluster SS (norm, UMAP)',
                        color=bar_colors_wcss, alpha=0.7)  # Use mapped colors
        bars2 = ax6_twin.bar(x_pos + 0.2, cluster_silhouettes[:actual_n_clusters], 0.4,
                             label='Avg Silhouette (UMAP)', color=[c for i, c in enumerate(bar_colors_sil)],
                             alpha=0.5)  # Darker or different shade

        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Normalized WCSS (UMAP)', color='blue')
        ax6_twin.set_ylabel('Average Silhouette Score (UMAP)', color='orange')
        ax6.set_title('Cluster Quality Metrics (on UMAP)', fontweight='bold')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([str(l) for l in unique_labels])  # Use actual cluster labels

        ax6.legend(loc='upper left')
        ax6_twin.legend(loc='upper right')
    else:
        ax6.text(0.5, 0.5, 'Metrics not applicable\n(No clusters found)',
                 horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)
        ax6.set_title('Cluster Quality Metrics (Skipped)', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.show()

    # Print cluster analysis
    print("=" * 60)
    print("UMAP + AGGLOMERATIVE CLUSTERING ANALYSIS")
    print("=" * 60)
    print(f"Target Number of Clusters: {n_clusters}")
    print(f"Actual Number of Clusters Found: {actual_n_clusters}")
    if silhouette_avg != -1:
        print(f"Overall Silhouette Score (on UMAP data): {silhouette_avg:.3f}")
    else:
        print("Overall Silhouette Score: Not computed.")
    print(f"Total Customers: {len(cluster_labels)}")
    print("\nCluster Distribution:")

    for cluster_k_label in unique_labels:
        cluster_size = np.sum(cluster_labels == cluster_k_label)
        if cluster_size > 0:
            percentage = (cluster_size / len(cluster_labels)) * 100
            if silhouette_avg != -1 and actual_n_clusters > 1:
                cluster_silhouette_val = sample_silhouette_values[cluster_labels == cluster_k_label].mean()
                print(f"  Cluster {cluster_k_label}: {cluster_size:4d} customers ({percentage:5.1f}%) - "
                      f"Avg Silhouette: {cluster_silhouette_val:.3f}")
            else:
                print(f"  Cluster {cluster_k_label}: {cluster_size:4d} customers ({percentage:5.1f}%)")
        # No need for an else for empty clusters if iterating unique_labels

    return X_umap, cluster_labels, silhouette_avg


def create_business_segment_profiles(df_original_features, cluster_labels, top_features=10):
    """
    Create business-interpretable segment profiles using original features.
    df_original_features: DataFrame with original (or enhanced pre-selection/scaling) features, indexed by card_id.
    cluster_labels: Array of cluster labels, aligned with df_original_features.
    """
    print(f"\nCreating business segment profiles...")

    profiling_df = df_original_features.copy()
    if len(profiling_df) != len(cluster_labels):
        print(f"Warning: Length of df_original_features ({len(profiling_df)}) does not match "
              f"length of cluster_labels ({len(cluster_labels)}). Profiles might be incorrect.")
        # Consider adding more robust alignment or error handling if this warning appears.
    profiling_df['cluster'] = cluster_labels

    numerical_features_df = profiling_df.select_dtypes(include=[np.number])

    # --- BEGIN FIX ---
    # Ensure columns are unique before calculating means to prevent ratio from being a Series.
    if numerical_features_df.columns.has_duplicates:
        duplicate_cols = numerical_features_df.columns[numerical_features_df.columns.duplicated()].tolist()
        print(f"Warning: Duplicate numerical feature names found in profiling data: {list(set(duplicate_cols))}. "
              "Keeping first occurrences for mean/ratio calculations.")
        numerical_features_df = numerical_features_df.loc[:, ~numerical_features_df.columns.duplicated(keep='first')]
    # --- END FIX ---

    # overall_means will now have a unique index.
    overall_means = numerical_features_df.drop(columns=['cluster'], errors='ignore').mean()

    cluster_profiles_summary = []
    unique_clusters = np.unique(cluster_labels)
    n_total_customers = len(cluster_labels)

    for cluster_id in unique_clusters:
        # cluster_data_numerical will inherit unique columns from the de-duplicated numerical_features_df.
        cluster_data_numerical = numerical_features_df[profiling_df['cluster'] == cluster_id]

        if cluster_data_numerical.empty:  # Should ideally not happen if cluster has members
            print(f"Warning: Cluster {cluster_id} has no numerical data, skipping profile.")
            continue

        cluster_size = len(cluster_data_numerical)  # Recalculate size from numerical data or use original count
        # For consistency, let's use the size from profiling_df which includes all data for the cluster
        original_cluster_size = np.sum(profiling_df['cluster'] == cluster_id)

        profile = {
            'Cluster': cluster_id,
            'Size': original_cluster_size,  # Use original count for size
            'Percentage': (original_cluster_size / n_total_customers) * 100
        }

        # feature_means_cluster will have a unique index.
        feature_means_cluster = cluster_data_numerical.drop(columns=['cluster'], errors='ignore').mean()

        # feature_ratios.index will be unique, so feature_ratios[feature] will be scalar.
        feature_ratios = feature_means_cluster / (overall_means + 1e-9)

        distinguishing_strength = (feature_ratios - 1).abs().sort_values(ascending=False)
        top_dist_features = distinguishing_strength.head(top_features).index

        for feature in top_dist_features:
            # Check if feature still exists after potential de-duplication and in current means
            if feature in feature_means_cluster and feature in feature_ratios:
                profile[f'{feature}_mean'] = feature_means_cluster[feature]
                # feature_ratios[feature] is now guaranteed to be a scalar.
                profile[f'{feature}_ratio_vs_overall'] = feature_ratios[feature]

        cluster_profiles_summary.append(profile)

    profiles_summary_df = pd.DataFrame(cluster_profiles_summary)

    print("\nCluster Profiles Summary (Top Distinguishing Features vs Overall Mean):")
    for _, row in profiles_summary_df.iterrows():  # row is a Series
        print(f"\nCluster {int(row['Cluster'])}: {int(row['Size'])} customers ({row['Percentage']:.1f}%)")

        dist_features_info = []
        for col_name in row.index:  # col_name is a string (column name)
            if col_name.endswith('_ratio_vs_overall'):
                feature_name = col_name.replace('_ratio_vs_overall', '')
                ratio = row[col_name]  # ratio should now be a scalar
                mean_val = row.get(f'{feature_name}_mean', 'N/A')

                # This condition should now work correctly as ratio is a scalar.
                if pd.notna(ratio):
                    dist_features_info.append((feature_name, ratio, mean_val, abs(ratio - 1)))

        dist_features_info.sort(key=lambda x: x[3], reverse=True)

        if dist_features_info:
            print("  Key Characteristics (Feature: Cluster Mean | Ratio to Overall):")
            for feature, ratio_val, mean_val, _ in dist_features_info[:top_features]:
                direction = "higher" if ratio_val > 1.01 else ("lower" if ratio_val < 0.99 else "similar")
                mean_str = f"{mean_val:.2f}" if isinstance(mean_val, (float, int)) else str(mean_val)
                print(f"    - {feature}: {mean_str} ({ratio_val:.2f}x, {direction} than average)")
        else:
            print("    No distinct numerical features found for this cluster based on ratios (or all ratios were NaN).")

    return profiles_summary_df

def find_optimal_clusters(X_processed, max_clusters=25):  # Increased max_clusters
    """Find optimal number of clusters using silhouette analysis on UMAP-reduced data"""
    print(f"\nFinding optimal number of clusters (up to {max_clusters})...")

    # Apply UMAP first (as clustering is done on UMAP data in the main analysis)
    print("Applying UMAP for optimal cluster search...")
    umap_reducer_search = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
    X_umap_search = umap_reducer_search.fit_transform(X_processed)
    print(f"UMAP for search completed: {X_umap_search.shape}")

    silhouette_scores = []
    # AgglomerativeClustering requires n_clusters >= 2
    cluster_range = range(2, max_clusters + 1)

    # Ensure we have enough samples for the smallest cluster number
    if len(X_umap_search) < 2:
        print("Not enough samples to perform clustering for optimal cluster search.")
        return 2, X_umap_search  # Default to 2, or handle error

    for n_c in cluster_range:
        if n_c > len(X_umap_search):  # Cannot have more clusters than samples
            print(f"  Skipping {n_c} clusters (more than sample size {len(X_umap_search)}).")
            silhouette_scores.append(-1)  # Invalid score
            continue
        try:
            clusterer = AgglomerativeClustering(n_clusters=n_c, linkage='ward')
            labels = clusterer.fit_predict(X_umap_search)
            if len(np.unique(labels)) > 1:  # Silhouette score requires at least 2 labels
                score = silhouette_score(X_umap_search, labels)
                silhouette_scores.append(score)
                print(f"  {n_c} clusters: Silhouette score = {score:.3f}")
            else:
                print(f"  {n_c} clusters: Only 1 cluster found. Silhouette not applicable.")
                silhouette_scores.append(-1)  # Or some other placeholder for not applicable
        except Exception as e:
            print(f"  {n_c} clusters: Error - {e}")
            silhouette_scores.append(-1)  # Error score

    # Filter out error scores for plotting and finding best
    valid_scores = [(n, s) for n, s in zip(cluster_range, silhouette_scores) if s > -1]

    if not valid_scores:
        print("No valid silhouette scores found for any cluster number.")
        return 2, X_umap_search  # Default or error

    valid_cluster_numbers, valid_silhouette_values = zip(*valid_scores)

    plt.figure(figsize=(10, 6))
    plt.plot(valid_cluster_numbers, valid_silhouette_values, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score (on UMAP data)')
    plt.title('Silhouette Analysis for Optimal Clusters (Agglomerative on UMAP)')
    plt.grid(True, alpha=0.3)

    if valid_silhouette_values:
        best_n_clusters_idx = np.argmax(valid_silhouette_values)
        best_n_clusters = valid_cluster_numbers[best_n_clusters_idx]
        best_score = valid_silhouette_values[best_n_clusters_idx]
        plt.axvline(x=best_n_clusters, color='red', linestyle='--', alpha=0.7)
        plt.text(best_n_clusters, best_score + 0.01, f'Best: {best_n_clusters} ({best_score:.3f})',
                 ha='center', color='red', fontweight='bold')
        print(f"\nOptimal number of clusters based on Silhouette (UMAP): {best_n_clusters} (Score: {best_score:.3f})")
    else:
        best_n_clusters = 2  # Default if no valid scores
        print("\nCould not determine optimal number of clusters from Silhouette scores.")

    plt.tight_layout()
    plt.show()

    return best_n_clusters, X_umap_search  # Return X_umap_search as it's already computed


def main():
    """Main execution function"""
    print("Starting clustering analysis...")
    N_CLUSTERS_TARGET = 14  # Target number of clusters as requested

    # Load and prepare data using functions from data_loader and feature_engineering
    # df_behavioral will have card_id as index
    df_behavioral, transaction_data = load_and_prepare_data()
    if df_behavioral is None or df_behavioral.empty:
        print("Exiting due to data loading or feature creation failure.")
        return

    # Prepare features for clustering
    print("\nPreparing features for clustering...")
    # Ensure we are using the DataFrame with card_id as index
    # And select only numerical features for the subsequent steps.
    X = df_behavioral.select_dtypes(include=[np.number]).copy()  # .copy() to avoid SettingWithCopyWarning later

    if X.empty:
        print("No numerical features found after initial creation!")
        return
    print(f"Initial numerical features: {len(X.columns)} for {len(X)} customers.")
    print(f"Feature names before selection: {X.columns.tolist()}")

    # 1. Remove low variance features
    print("\nStep 1: Removing low variance features...")
    X = remove_low_variance_features(X, threshold=0.01)  # Use default or adjust threshold
    print(f"Features after low variance removal: {len(X.columns)}. Names: {X.columns.tolist()}")
    if X.empty:
        print("No features remaining after low variance removal!")
        return

    # 2. Remove highly correlated features
    print("\nStep 2: Removing highly correlated features...")
    X = remove_highly_correlated_features(X, threshold=0.95)  # Use default or adjust threshold
    print(f"Features after high correlation removal: {len(X.columns)}. Names: {X.columns.tolist()}")
    if X.empty:
        print("No features remaining after high correlation removal!")
        return

    feature_names = X.columns.tolist()  # Update feature names after selection

    # 3. Preprocess features (outlier handling and scaling)
    print("\nStep 3: Preprocessing features (outliers and scaling)...")
    # preprocess_features_for_clustering returns a tuple: (processed_df, scaler_object or None)
    # We need the processed DataFrame for clustering.
    X_processed, scaler_object = preprocess_features_for_clustering(
        X.copy(),  # Pass a copy to avoid modifying X if it's used elsewhere
        remove_outliers=True,  # As per your interface, default is True
        scale_features=True  # Essential for UMAP/distance-based clustering
    )
    if X_processed.empty:
        print("No features remaining after preprocessing!")
        return
    print(
        f"Features after preprocessing: {len(X_processed.columns)} (should be same as after correlation removal if no columns were all NaNs).")
    print(f"Data shape for clustering: {X_processed.shape}")

    # Set pandas display options for better output if needed for debugging or inspection
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)  # Adjusted width

    # Find optimal number of clusters (for informational purposes)
    # The X_umap_search from here is specific to this search, main analysis will re-run UMAP
    # We pass X_processed which is scaled and feature-selected.
    optimal_clusters_info, _ = find_optimal_clusters(X_processed.copy(), max_clusters=25)  # Use copy
    print(f"\nInformational: Optimal clusters suggested by Silhouette: {optimal_clusters_info}")
    print(f"Proceeding with the requested {N_CLUSTERS_TARGET} clusters for detailed analysis.")

    # Perform detailed analysis with the *requested* number of clusters (14)
    # analyze_umap_agglomerative will perform UMAP on X_processed
    # feature_names here are from X (after selection, before scaling by preprocess_features_for_clustering)
    # However, analyze_umap_agglomerative itself doesn't use feature_names directly for calcs, mostly for UMAP.
    # UMAP operates on the numerical data X_processed.
    final_X_umap, final_cluster_labels, final_silhouette_score = analyze_umap_agglomerative(
        X_processed.values,  # Pass numpy array to UMAP/clustering as is common
        feature_names,
        # Original feature names for context if needed by plotting (though current plots use UMAP components)
        n_clusters=N_CLUSTERS_TARGET
    )

    # Create business profiles using the original behavioral features (df_behavioral)
    # and the final cluster labels. df_behavioral has card_id as index.
    # The cluster labels are in the same order as the rows of X_processed (and X, and df_behavioral).
    if final_cluster_labels is not None and len(final_cluster_labels) == len(df_behavioral):
        profiles_df = create_business_segment_profiles(df_behavioral, final_cluster_labels, top_features=7)
    else:
        print("Could not create business profiles due to mismatch in lengths or no cluster labels.")
        profiles_df = pd.DataFrame()

    # Save results
    if final_cluster_labels is not None and len(final_cluster_labels) == len(df_behavioral):
        try:
            # Create results dataframe by adding cluster and UMAP components to the behavioral df (indexed by card_id)
            results_df = df_behavioral.copy()  # df_behavioral already has card_id as index
            results_df['cluster'] = final_cluster_labels
            results_df['umap_1'] = final_X_umap[:, 0]
            results_df['umap_2'] = final_X_umap[:, 1]

            print(f"\nSaving results...")
            results_df.to_csv('clustering_results_14_clusters.csv', index=True)  # Index is card_id
            if not profiles_df.empty:
                profiles_df.to_csv('cluster_profiles_14_clusters.csv', index=False)
            print("Results saved to 'clustering_results_14_clusters.csv' and 'cluster_profiles_14_clusters.csv'")

        except Exception as e:
            print(f"Error saving results: {e}")
    else:
        print("\nSkipping saving results as cluster assignments are not available or mismatched.")

    print("\nClustering analysis completed!")
    # Return for potential further use if run in an interactive environment
    # return results_df if 'results_df' in locals() else None, profiles_df


if __name__ == "__main__":
    # Execute main analysis
    main()