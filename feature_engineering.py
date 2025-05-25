import pandas as pd
import numpy as np
from typing import Optional
from icontract import require, ensure
from sklearn.preprocessing import StandardScaler
import warnings


@require(lambda df: df is None or isinstance(df, pd.DataFrame))
@ensure(lambda result: isinstance(result, pd.DataFrame) and (result.empty or result.index.name == 'card_id'))
def create_enhanced_behavioral_features(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Creates comprehensive behavioral features for better customer segmentation.

    :param df: The transaction data.
    :return: DataFrame with card_id as index and enhanced behavioral features as columns.
    """
    if df is None or df.empty:
        return pd.DataFrame(index=pd.Index([], name='card_id'))

    # Data validation and preprocessing
    df = df.copy()

    # Handle timestamp conversion with proper error handling
    try:
        df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
    except Exception as e:
        print(f"Warning: Could not convert transaction_timestamp: {e}")
        # Create a dummy timestamp if conversion fails
        df['transaction_timestamp'] = pd.Timestamp.now()

    df = df.sort_values(by=['card_id', 'transaction_timestamp'])

    # Extract time components
    df['hour'] = df['transaction_timestamp'].dt.hour
    df['day_of_week'] = df['transaction_timestamp'].dt.dayofweek
    df['month'] = df['transaction_timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Time differences between consecutive transactions
    df['time_diff'] = df.groupby('card_id')['transaction_timestamp'].diff().dt.total_seconds()

    feature_list = []

    # === VOLUME FEATURES ===
    n_txn = df.groupby('card_id').size().rename('num_transactions')
    feature_list.append(n_txn)

    total_amount = df.groupby('card_id')['transaction_amount_kzt'].sum().rename('total_spend')
    feature_list.append(total_amount)

    # === TEMPORAL BEHAVIOR FEATURES ===

    # 1. Activity concentration - how spread out is their activity?
    active_days = df.groupby('card_id')['transaction_timestamp'].apply(
        lambda x: x.dt.date.nunique()
    ).rename('active_days')

    # Safe division with zero handling
    activity_intensity = (n_txn / active_days.replace(0, 1)).rename('txns_per_active_day')
    feature_list.extend([active_days, activity_intensity])

    # 2. Weekend vs weekday behavior
    weekend_txn_ratio = df.groupby('card_id')['is_weekend'].mean().rename('weekend_txn_ratio')
    feature_list.append(weekend_txn_ratio)

    weekend_spend_ratio = df.groupby('card_id').apply(
        lambda x: x[x['is_weekend'] == 1]['transaction_amount_kzt'].sum() /
                  max(x['transaction_amount_kzt'].sum(), 1e-10)  # Avoid division by zero
    ).rename('weekend_spend_ratio')
    feature_list.append(weekend_spend_ratio)

    # 3. Time of day patterns
    business_hours_ratio = df.groupby('card_id').apply(
        lambda x: ((x['hour'] >= 9) & (x['hour'] <= 17)).mean()
    ).rename('business_hours_ratio')
    feature_list.append(business_hours_ratio)

    evening_ratio = df.groupby('card_id').apply(
        lambda x: ((x['hour'] >= 18) & (x['hour'] <= 22)).mean()
    ).rename('evening_txn_ratio')
    feature_list.append(evening_ratio)

    night_ratio = df.groupby('card_id').apply(
        lambda x: ((x['hour'] >= 23) | (x['hour'] <= 5)).mean()
    ).rename('night_txn_ratio')
    feature_list.append(night_ratio)

    # 4. Regularity patterns - improved with better handling
    time_between_txns_cv = df.groupby('card_id')['time_diff'].apply(
        lambda x: x.std() / max(x.mean(), 1e-10) if len(x) > 1 and x.mean() > 0 else 0
    ).fillna(0).rename('time_regularity_cv')
    feature_list.append(time_between_txns_cv)

    # === SPENDING BEHAVIOR FEATURES ===

    # 1. Transaction size patterns
    mean_txn_amount = df.groupby('card_id')['transaction_amount_kzt'].mean().rename('avg_txn_amount')
    median_txn_amount = df.groupby('card_id')['transaction_amount_kzt'].median().rename('median_txn_amount')

    txn_amount_cv = df.groupby('card_id')['transaction_amount_kzt'].apply(
        lambda x: x.std() / max(x.mean(), 1e-10) if x.mean() > 0 else 0
    ).fillna(0).rename('txn_amount_variability')
    feature_list.extend([mean_txn_amount, median_txn_amount, txn_amount_cv])

    # 2. Large transaction behavior (use 95th percentile as threshold)
    large_txn_threshold = df['transaction_amount_kzt'].quantile(0.95)
    large_txn_ratio = df.groupby('card_id').apply(
        lambda x: (x['transaction_amount_kzt'] > large_txn_threshold).mean()
    ).rename('large_txn_ratio')

    large_txn_amount = df.groupby('card_id').apply(
        lambda x: x[x['transaction_amount_kzt'] > large_txn_threshold]['transaction_amount_kzt'].sum()
    ).rename('large_txn_total_amount')
    feature_list.extend([large_txn_ratio, large_txn_amount])

    # 3. Small transaction behavior (micro-transactions)
    small_txn_threshold = df['transaction_amount_kzt'].quantile(0.05)
    small_txn_ratio = df.groupby('card_id').apply(
        lambda x: (x['transaction_amount_kzt'] < small_txn_threshold).mean()
    ).rename('small_txn_ratio')
    feature_list.append(small_txn_ratio)

    # === MERCHANT & CATEGORY DIVERSITY ===

    # 1. Merchant loyalty (handle null merchant_id)
    df_with_merchant = df[df['merchant_id'].notna()]
    if not df_with_merchant.empty:
        unique_merchants = df_with_merchant.groupby('card_id')['merchant_id'].nunique().rename('unique_merchants')
        txn_with_merchant = df_with_merchant.groupby('card_id').size().rename('txn_with_merchant')
        txn_per_merchant = (txn_with_merchant / unique_merchants.replace(0, 1)).rename('txns_per_merchant')

        # Top merchant concentration
        top_merchant_ratio = df_with_merchant.groupby('card_id').apply(
            lambda x: x['merchant_id'].value_counts().iloc[0] / len(x) if len(x) > 0 else 0
        ).rename('top_merchant_concentration')

        feature_list.extend([unique_merchants, txn_per_merchant, top_merchant_ratio])
    else:
        # Create dummy features if no merchant data
        dummy_merchant_features = pd.Series(0, index=n_txn.index, name='unique_merchants')
        feature_list.append(dummy_merchant_features)

    # 2. Category diversity (MCC) - handle potential missing values
    if 'merchant_mcc' in df.columns:
        df_with_mcc = df[df['merchant_mcc'].notna()]
        if not df_with_mcc.empty:
            unique_mccs = df_with_mcc.groupby('card_id')['merchant_mcc'].nunique().rename('unique_categories')
            txn_with_mcc = df_with_mcc.groupby('card_id').size().rename('txn_with_mcc')
            category_diversity = (unique_mccs / txn_with_mcc.replace(0, 1)).rename('category_diversity')

            # MCC concentration patterns
            top_mcc_concentration = df_with_mcc.groupby('card_id').apply(
                lambda x: x['merchant_mcc'].value_counts().iloc[0] / len(x) if len(x) > 0 else 0
            ).rename('top_category_concentration')

            feature_list.extend([unique_mccs, category_diversity, top_mcc_concentration])
        else:
            # Create dummy features if no MCC data
            dummy_mcc_features = pd.Series(0, index=n_txn.index, name='unique_categories')
            feature_list.append(dummy_mcc_features)

    # === PAYMENT METHOD PREFERENCES ===

    # 1. Transaction type preferences - handle different transaction types dynamically
    available_txn_types = df['transaction_type'].dropna().unique()

    for txn_type in available_txn_types:
        safe_name = txn_type.lower().replace('_', '').replace('-', '')
        feature_name = f'{safe_name}_preference'

        txn_type_ratio = df.groupby('card_id').apply(
            lambda x: (x['transaction_type'] == txn_type).mean()
        ).rename(feature_name)
        feature_list.append(txn_type_ratio)

    # 2. POS entry mode preferences - handle if available
    if 'pos_entry_mode' in df.columns:
        df_pos = df[df['pos_entry_mode'].notna()]
        if not df_pos.empty:
            available_entry_modes = df_pos['pos_entry_mode'].unique()

            for entry_mode in available_entry_modes:
                safe_name = entry_mode.lower().replace(' ', '')
                feature_name = f'{safe_name}_preference'

                entry_mode_ratio = df.groupby('card_id').apply(
                    lambda x: (x['pos_entry_mode'] == entry_mode).mean()
                ).rename(feature_name)
                feature_list.append(entry_mode_ratio)

    # 3. Digital wallet usage - handle if available
    if 'wallet_type' in df.columns:
        has_wallet = df.groupby('card_id')['wallet_type'].apply(
            lambda x: x.notna().any()
        ).astype(int).rename('uses_digital_wallet')
        feature_list.append(has_wallet)

        # Wallet diversity
        wallet_diversity = df[df['wallet_type'].notna()].groupby('card_id')['wallet_type'].nunique().fillna(0).rename(
            'wallet_type_diversity')
        feature_list.append(wallet_diversity)

    # === GEOGRAPHIC BEHAVIOR ===

    # 1. Multi-currency usage - handle if available
    if 'transaction_currency' in df.columns:
        multi_currency_user = df.groupby('card_id')['transaction_currency'].apply(
            lambda x: (x != 'KZT').any() if x.notna().any() else False
        ).astype(int).rename('uses_foreign_currency')
        feature_list.append(multi_currency_user)

        # Currency diversity
        currency_diversity = df.groupby('card_id')['transaction_currency'].nunique().rename('currency_diversity')
        feature_list.append(currency_diversity)

    # 2. Foreign transaction patterns - handle if available
    if 'acquirer_country_iso' in df.columns:
        foreign_txn_ratio = df.groupby('card_id').apply(
            lambda x: (x['acquirer_country_iso'] != 'KAZ').mean() if x['acquirer_country_iso'].notna().any() else 0
        ).rename('foreign_txn_ratio')
        feature_list.append(foreign_txn_ratio)

        # Geographic diversity
        unique_countries = df.groupby('card_id')['acquirer_country_iso'].nunique().rename('countries_visited')
        feature_list.append(unique_countries)

    # 3. City diversity (if available)
    if 'merchant_city' in df.columns:
        df_with_city = df[df['merchant_city'].notna()]
        if not df_with_city.empty:
            unique_cities = df_with_city.groupby('card_id')['merchant_city'].nunique().rename('unique_cities')
            feature_list.append(unique_cities)

    # === CARD LIFECYCLE FEATURES ===

    # 1. Card expiry handling - improved date parsing
    if 'expiry_date' in df.columns:
        try:
            # Handle different expiry date formats
            df['expiry_date_parsed'] = pd.to_datetime(df['expiry_date'], format='%m/%y', errors='coerce')

            # If that fails, try other common formats
            mask_failed = df['expiry_date_parsed'].isna()
            if mask_failed.any():
                df.loc[mask_failed, 'expiry_date_parsed'] = pd.to_datetime(
                    df.loc[mask_failed, 'expiry_date'], format='%m/%Y', errors='coerce'
                )

            current_date = pd.Timestamp.now()
            card_remaining_months = df.groupby('card_id')['expiry_date_parsed'].first().apply(
                lambda x: max((x - current_date).days / 30, 0) if pd.notna(x) else 0
            ).rename('card_remaining_months')
            feature_list.append(card_remaining_months)

        except Exception as e:
            print(f"Warning: Could not process expiry_date: {e}")

    # 2. Usage recency and activity patterns
    current_time = pd.Timestamp.now()
    last_txn_days_ago = df.groupby('card_id')['transaction_timestamp'].apply(
        lambda x: max((current_time - x.max()).days, 0)
    ).rename('days_since_last_txn')

    first_txn_days_ago = df.groupby('card_id')['transaction_timestamp'].apply(
        lambda x: max((current_time - x.min()).days, 0)
    ).rename('days_since_first_txn')

    feature_list.extend([last_txn_days_ago, first_txn_days_ago])

    # === BEHAVIORAL RATIOS AND DERIVED FEATURES ===

    # 1. Spend per day ratio - improved calculation
    date_range = df.groupby('card_id')['transaction_timestamp'].apply(
        lambda x: max((x.max() - x.min()).days, 1)  # Ensure at least 1 day
    ).rename('observation_days')

    daily_spend = (total_amount / date_range).rename('avg_daily_spend')
    feature_list.extend([date_range, daily_spend])

    # 2. Transaction frequency consistency
    daily_txn_counts = df.groupby(['card_id', df['transaction_timestamp'].dt.date]).size()
    txn_frequency_cv = daily_txn_counts.groupby('card_id').apply(
        lambda x: x.std() / max(x.mean(), 1e-10) if x.mean() > 0 and len(x) > 1 else 0
    ).fillna(0).rename('daily_txn_consistency')
    feature_list.append(txn_frequency_cv)

    # 3. Additional derived features
    # Amount per transaction ratios
    max_txn_amount = df.groupby('card_id')['transaction_amount_kzt'].max().rename('max_txn_amount')
    min_txn_amount = df.groupby('card_id')['transaction_amount_kzt'].min().rename('min_txn_amount')
    amount_range = (max_txn_amount - min_txn_amount).rename('txn_amount_range')

    feature_list.extend([max_txn_amount, min_txn_amount, amount_range])

    # Frequency patterns
    monthly_txn_count = df.groupby(['card_id', 'month']).size().groupby('card_id').mean().rename('avg_monthly_txns')
    feature_list.append(monthly_txn_count)

    # === COMBINE ALL FEATURES ===
    try:
        # Ensure all features have the same index
        common_index = n_txn.index
        aligned_features = []

        for feature in feature_list:
            if isinstance(feature, pd.Series):
                # Reindex to common index, filling missing values with 0
                aligned_feature = feature.reindex(common_index, fill_value=0)
                aligned_features.append(aligned_feature)

        features = pd.concat(aligned_features, axis=1)
        features.index.name = 'card_id'

        # Fill any remaining NaN values with 0
        features = features.fillna(0)

        # Handle infinite values
        features = features.replace([np.inf, -np.inf], 0)

        return features

    except Exception as e:
        print(f"Error combining features: {e}")
        # Return minimal feature set in case of error
        basic_features = pd.concat([n_txn, total_amount], axis=1)
        basic_features.index.name = 'card_id'
        return basic_features.fillna(0)


def remove_low_variance_features(X, threshold=0.01):
    """Remove features with very low variance that don't contribute to segmentation"""
    from sklearn.feature_selection import VarianceThreshold

    if X.empty:
        return X

    try:
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()]

        print(f"Removed {len(X.columns) - len(selected_features)} low-variance features")
        print(f"Remaining features: {len(selected_features)}")

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    except Exception as e:
        print(f"Warning: Could not apply variance threshold: {e}")
        return X


def remove_highly_correlated_features(X, threshold=0.95):
    """Remove features that are highly correlated with others"""

    if X.empty or len(X.columns) < 2:
        return X

    try:
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

        print(f"Removing {len(to_drop)} highly correlated features (correlation > {threshold})")

        return X.drop(columns=to_drop)

    except Exception as e:
        print(f"Warning: Could not remove correlated features: {e}")
        return X


def create_behavioral_segments_analysis(df, features):
    """Analyze the created features to suggest good segmentation approaches"""

    print("=== FEATURE ANALYSIS FOR SEGMENTATION ===")
    print(f"Total features created: {len(features.columns)}")
    print(f"Total customers: {len(features)}")

    if features.empty:
        print("No features to analyze")
        return {}

    # Feature correlation analysis
    try:
        corr_matrix = features.corr()
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:  # High correlation threshold
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))

        if high_corr_pairs:
            print(f"\nHigh correlation pairs (>0.8): {len(high_corr_pairs)}")
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:5]:
                print(f"  {feat1} <-> {feat2}: {corr:.3f}")
        else:
            print("\nNo highly correlated feature pairs found")

    except Exception as e:
        print(f"Warning: Could not analyze feature correlations: {e}")

    # Feature variance analysis
    try:
        feature_vars = features.var().sort_values(ascending=False)
        print(f"\nTop 10 most variable features:")
        for feat, var in feature_vars.head(10).items():
            print(f"  {feat}: {var:.2e}")

        print(f"\nBottom 5 least variable features:")
        for feat, var in feature_vars.tail(5).items():
            print(f"  {feat}: {var:.2e}")

    except Exception as e:
        print(f"Warning: Could not analyze feature variance: {e}")

    # Suggest feature groups for interpretation
    temporal_features = [col for col in features.columns if any(term in col.lower()
                                                                for term in
                                                                ['weekend', 'evening', 'business', 'time', 'day',
                                                                 'regular', 'night', 'hour'])]
    spending_features = [col for col in features.columns if any(term in col.lower()
                                                                for term in
                                                                ['amount', 'spend', 'large', 'small', 'txn'])]
    diversity_features = [col for col in features.columns if any(term in col.lower()
                                                                 for term in
                                                                 ['unique', 'merchant', 'category', 'diversity',
                                                                  'concentration', 'mcc'])]
    payment_features = [col for col in features.columns if any(term in col.lower()
                                                               for term in
                                                               ['ecom', 'pos', 'contact', 'chip', 'wallet', 'atm',
                                                                'preference'])]
    geo_features = [col for col in features.columns if any(term in col.lower()
                                                           for term in ['foreign', 'country', 'city', 'currency'])]

    print(f"\nFeature categories:")
    print(f"  Temporal behavior: {len(temporal_features)} features")
    print(f"  Spending patterns: {len(spending_features)} features")
    print(f"  Merchant/Category diversity: {len(diversity_features)} features")
    print(f"  Payment preferences: {len(payment_features)} features")
    print(f"  Geographic behavior: {len(geo_features)} features")

    return {
        'temporal': temporal_features,
        'spending': spending_features,
        'diversity': diversity_features,
        'payment': payment_features,
        'geographic': geo_features
    }


def preprocess_features_for_clustering(features, remove_outliers=True, scale_features=True):
    """Preprocess features for clustering by handling outliers and scaling"""

    if features.empty:
        return features, None

    processed_features = features.copy()

    # Remove outliers using IQR method
    if remove_outliers:
        print("Removing outliers...")
        for column in processed_features.columns:
            Q1 = processed_features[column].quantile(0.25)
            Q3 = processed_features[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap outliers instead of removing rows to preserve all customers
            processed_features[column] = processed_features[column].clip(lower_bound, upper_bound)

    # Scale features
    scaler = None
    if scale_features:
        print("Scaling features...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(processed_features)
        processed_features = pd.DataFrame(
            scaled_data,
            columns=processed_features.columns,
            index=processed_features.index
        )

    return processed_features, scaler


# Usage example
if __name__ == '__main__':
    # Note: This assumes you have a data_loader module
    try:
        from data_loader import load_transaction_data

        transaction_data = load_transaction_data('data/transactions.parquet')
        if transaction_data is not None and not transaction_data.empty:
            print(
                f"Loaded {len(transaction_data)} transactions for {transaction_data['card_id'].nunique()} unique cards")

            # Create enhanced features
            enhanced_features = create_enhanced_behavioral_features(transaction_data)

            if not enhanced_features.empty:
                print(f"Created {len(enhanced_features.columns)} features for {len(enhanced_features)} customers")

                # Remove low variance and highly correlated features
                enhanced_features_clean = remove_low_variance_features(enhanced_features)
                enhanced_features_clean = remove_highly_correlated_features(enhanced_features_clean)

                # Analyze features
                feature_groups = create_behavioral_segments_analysis(transaction_data, enhanced_features_clean)

                # Preprocess for clustering
                processed_features, scaler = preprocess_features_for_clustering(enhanced_features_clean)

                print(f"\nFinal feature set shape: {processed_features.shape}")
                print("\nSample of features:")
                print(processed_features.head())

                # Display feature summary
                print(f"\nFeature summary:")
                print(processed_features.describe())

            else:
                print("No features were created")
        else:
            print("No transaction data loaded")

    except ImportError:
        print("data_loader module not found. Please ensure the data loading module is available.")
    except Exception as e:
        print(f"Error in feature engineering: {e}")