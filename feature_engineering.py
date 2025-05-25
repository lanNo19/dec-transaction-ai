import pandas as pd
import numpy as np
from typing import Optional
from icontract import require, ensure
from sklearn.preprocessing import StandardScaler


@require(lambda df: df is None or isinstance(df, pd.DataFrame))
@ensure(lambda result: isinstance(result, pd.DataFrame) and (result.empty or result.index.name == 'card_id'))
def create_enhanced_behavioral_features(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Creates comprehensive behavioral features for better customer segmentation.

    :param df: The transaction data.
    :return: DataFrame with card_id as index and enhanced behavioral features as columns.
    """

    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
    df = df.sort_values(by=['card_id', 'transaction_timestamp'])

    # Extract time components
    df['hour'] = df['transaction_timestamp'].dt.hour
    df['day_of_week'] = df['transaction_timestamp'].dt.dayofweek
    df['month'] = df['transaction_timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Time differences
    df['time_diff'] = df.groupby('card_id')['transaction_timestamp'].diff().dt.total_seconds()

    feature_list = []

    # === VOLUME FEATURES (keep essential ones) ===
    n_txn = df.groupby('card_id').size().rename('num_transactions')
    feature_list.append(n_txn)

    total_amount = df.groupby('card_id')['transaction_amount_kzt'].sum().rename('total_spend')
    feature_list.append(total_amount)

    # === TEMPORAL BEHAVIOR FEATURES ===

    # 1. Activity concentration - how spread out is their activity?
    active_days = df.groupby('card_id')['transaction_timestamp'].apply(
        lambda x: x.dt.date.nunique()
    ).rename('active_days')
    activity_intensity = (n_txn / active_days).rename('txns_per_active_day')
    feature_list.extend([active_days, activity_intensity])

    # 2. Weekend vs weekday behavior
    weekend_txn_ratio = df.groupby('card_id')['is_weekend'].mean().rename('weekend_txn_ratio')
    feature_list.append(weekend_txn_ratio)

    weekend_spend_ratio = df.groupby('card_id').apply(
        lambda x: x[x['is_weekend'] == 1]['transaction_amount_kzt'].sum() /
                  x['transaction_amount_kzt'].sum() if x['transaction_amount_kzt'].sum() > 0 else 0
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

    # 4. Regularity patterns
    time_between_txns_cv = df.groupby('card_id')['time_diff'].apply(
        lambda x: x.std() / x.mean() if x.mean() > 0 and len(x) > 1 else 0
    ).fillna(0).rename('time_regularity_cv')
    feature_list.append(time_between_txns_cv)

    # === SPENDING BEHAVIOR FEATURES ===

    # 1. Transaction size patterns
    mean_txn_amount = df.groupby('card_id')['transaction_amount_kzt'].mean().rename('avg_txn_amount')
    txn_amount_cv = df.groupby('card_id')['transaction_amount_kzt'].apply(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 0
    ).fillna(0).rename('txn_amount_variability')
    feature_list.extend([mean_txn_amount, txn_amount_cv])

    # 2. Large transaction behavior
    large_txn_threshold = df['transaction_amount_kzt'].quantile(0.90)
    large_txn_ratio = df.groupby('card_id').apply(
        lambda x: (x['transaction_amount_kzt'] > large_txn_threshold).mean()
    ).rename('large_txn_ratio')
    feature_list.append(large_txn_ratio)

    # 3. Small transaction behavior (micro-transactions)
    small_txn_threshold = df['transaction_amount_kzt'].quantile(0.10)
    small_txn_ratio = df.groupby('card_id').apply(
        lambda x: (x['transaction_amount_kzt'] < small_txn_threshold).mean()
    ).rename('small_txn_ratio')
    feature_list.append(small_txn_ratio)

    # === MERCHANT & CATEGORY DIVERSITY ===

    # 1. Merchant loyalty
    unique_merchants = df.groupby('card_id')['merchant_id'].nunique().rename('unique_merchants')
    txn_per_merchant = (n_txn / unique_merchants).rename('txns_per_merchant')
    feature_list.extend([unique_merchants, txn_per_merchant])

    # Top merchant concentration
    top_merchant_ratio = df.groupby('card_id').apply(
        lambda x: x['merchant_id'].value_counts().iloc[0] / len(x) if len(x) > 0 else 0
    ).rename('top_merchant_concentration')
    feature_list.append(top_merchant_ratio)

    # 2. Category diversity (MCC)
    unique_mccs = df.groupby('card_id')['merchant_mcc'].nunique().rename('unique_categories')
    category_diversity = unique_mccs / n_txn  # Categories per transaction
    category_diversity.name = 'category_diversity'
    feature_list.extend([unique_mccs, category_diversity])

    # 3. MCC concentration patterns
    top_mcc_concentration = df.groupby('card_id').apply(
        lambda x: x['merchant_mcc'].value_counts().iloc[0] / len(x) if len(x) > 0 else 0
    ).rename('top_category_concentration')
    feature_list.append(top_mcc_concentration)

    # === PAYMENT METHOD PREFERENCES ===

    # 1. Transaction type preferences
    ecom_ratio = df.groupby('card_id').apply(
        lambda x: (x['transaction_type'] == 'ECOM').mean()
    ).rename('ecom_preference')
    feature_list.append(ecom_ratio)

    pos_ratio = df.groupby('card_id').apply(
        lambda x: (x['transaction_type'] == 'POS').mean()
    ).rename('pos_preference')
    feature_list.append(pos_ratio)

    atm_ratio = df.groupby('card_id').apply(
        lambda x: (x['transaction_type'] == 'ATM_WITHDRAWAL').mean()
    ).rename('atm_preference')
    feature_list.append(atm_ratio)

    # 2. POS entry mode preferences
    contactless_ratio = df.groupby('card_id').apply(
        lambda x: (x['pos_entry_mode'] == 'Contactless').mean()
    ).rename('contactless_preference')
    feature_list.append(contactless_ratio)

    chip_ratio = df.groupby('card_id').apply(
        lambda x: (x['pos_entry_mode'] == 'Chip').mean()
    ).rename('chip_preference')
    feature_list.append(chip_ratio)

    # 3. Digital wallet usage
    has_wallet = df.groupby('card_id')['wallet_type'].apply(
        lambda x: x.notna().any()
    ).astype(int).rename('uses_digital_wallet')
    feature_list.append(has_wallet)

    # === GEOGRAPHIC BEHAVIOR ===

    # 1. Multi-currency usage
    multi_currency_user = df.groupby('card_id')['transaction_currency'].apply(
        lambda x: (x != 'KZT').any()
    ).astype(int).rename('uses_foreign_currency')
    feature_list.append(multi_currency_user)

    # 2. Foreign transaction patterns
    foreign_txn_ratio = df.groupby('card_id').apply(
        lambda x: (x['acquirer_country_iso'] != 'KAZ').mean()
    ).rename('foreign_txn_ratio')
    feature_list.append(foreign_txn_ratio)

    # 3. Geographic diversity
    unique_countries = df.groupby('card_id')['acquirer_country_iso'].nunique().rename('countries_visited')
    feature_list.append(unique_countries)

    # City diversity (if available)
    if 'merchant_city' in df.columns:
        unique_cities = df.groupby('card_id')['merchant_city'].nunique().rename('unique_cities')
        feature_list.append(unique_cities)

    # === CARD LIFECYCLE FEATURES ===

    # 1. Account age and usage
    df['expiry_date'] = pd.to_datetime(df['expiry_date'], format='%m/%y', errors='coerce')
    account_age_months = df.groupby('card_id')['expiry_date'].first().apply(
        lambda x: (x - pd.Timestamp.now()).days / 30 if pd.notna(x) else 0
    ).rename('card_remaining_months')
    feature_list.append(account_age_months)

    # 2. Usage recency and activity patterns
    last_txn_days_ago = df.groupby('card_id')['transaction_timestamp'].apply(
        lambda x: (pd.Timestamp.now() - x.max()).days
    ).rename('days_since_last_txn')
    feature_list.append(last_txn_days_ago)

    # === BEHAVIORAL RATIOS AND DERIVED FEATURES ===

    # 1. Spend per day ratio
    date_range = df.groupby('card_id')['transaction_timestamp'].apply(
        lambda x: (x.max() - x.min()).days + 1
    ).rename('observation_days')
    daily_spend = (total_amount / date_range).rename('avg_daily_spend')
    feature_list.extend([daily_spend])

    # 2. Transaction frequency consistency
    # Calculate variance in daily transaction counts
    daily_txn_counts = df.groupby(['card_id', df['transaction_timestamp'].dt.date]).size()
    txn_frequency_cv = daily_txn_counts.groupby('card_id').apply(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 0
    ).fillna(0).rename('daily_txn_consistency')
    feature_list.append(txn_frequency_cv)

    # Combine all features
    features = pd.concat(feature_list, axis=1)
    features.index.name = 'card_id'

    # Fill any remaining NaN values
    features = features.fillna(0)

    return features


def remove_low_variance_features(X, threshold=0.01):
    """Remove features with very low variance that don't contribute to segmentation"""
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()]

    print(f"Removed {len(X.columns) - len(selected_features)} low-variance features")
    print(f"Remaining features: {len(selected_features)}")

    return pd.DataFrame(X_selected, columns=selected_features, index=X.index)


def create_behavioral_segments_analysis(df, features):
    """Analyze the created features to suggest good segmentation approaches"""

    print("=== FEATURE ANALYSIS FOR SEGMENTATION ===")
    print(f"Total features created: {len(features.columns)}")
    print(f"Total customers: {len(features)}")

    # Feature correlation analysis
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

    # Feature variance analysis
    feature_vars = features.var().sort_values(ascending=False)
    print(f"\nTop 10 most variable features:")
    for feat, var in feature_vars.head(10).items():
        print(f"  {feat}: {var:.2e}")

    print(f"\nBottom 5 least variable features:")
    for feat, var in feature_vars.tail(5).items():
        print(f"  {feat}: {var:.2e}")

    # Suggest feature groups for interpretation
    temporal_features = [col for col in features.columns if any(term in col.lower()
                                                                for term in
                                                                ['weekend', 'evening', 'business', 'time', 'day',
                                                                 'regular'])]
    spending_features = [col for col in features.columns if any(term in col.lower()
                                                                for term in ['amount', 'spend', 'large', 'small'])]
    diversity_features = [col for col in features.columns if any(term in col.lower()
                                                                 for term in
                                                                 ['unique', 'merchant', 'category', 'diversity',
                                                                  'concentration'])]
    payment_features = [col for col in features.columns if any(term in col.lower()
                                                               for term in
                                                               ['ecom', 'pos', 'contact', 'chip', 'wallet', 'atm'])]
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


# Usage example
if __name__ == '__main__':
    from data_loader import load_transaction_data

    transaction_data = load_transaction_data('data/transactions.parquet')
    if transaction_data is not None:
        # Create enhanced features
        enhanced_features = create_enhanced_behavioral_features(transaction_data)

        # Remove low variance features
        enhanced_features_clean = remove_low_variance_features(enhanced_features)

        # Analyze features
        feature_groups = create_behavioral_segments_analysis(transaction_data, enhanced_features_clean)

        print(f"\nFinal feature set shape: {enhanced_features_clean.shape}")
        print("\nSample of features:")
        print(enhanced_features_clean.head())

    else:
        print("Failed to load transaction data")