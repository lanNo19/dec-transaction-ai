import pandas as pd
from typing import Optional
from icontract import require, ensure

@require(lambda df: df is None or isinstance(df, pd.DataFrame))
@ensure(lambda result: isinstance(result, pd.DataFrame) and (result.empty or result.index.name == 'card_id'))
def create_behavioral_features(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Creates behavioral features for each cardholder.

    :param df: The transaction data.
    :return: DataFrame with card_id as index and behavioral features as columns.
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
    df = df.sort_values(by=['card_id', 'transaction_timestamp'])
    df['time_diff'] = df.groupby('card_id')['transaction_timestamp'].diff().dt.total_seconds()

    # 1. Number of transactions
    n_txn = df.groupby('card_id').size().rename('num_transactions')

    # 2-4. Mean, Std, and ratio of time_diff
    time_diff_stats = df.groupby('card_id')['time_diff'].agg([
        ('mean_time_diff', 'mean'),
        ('std_time_diff', 'std')
    ])

    # Fill NaN std with 0
    time_diff_stats['std_time_diff'] = time_diff_stats['std_time_diff'].fillna(0)

    # Compute ratio and fill NaN (e.g., if mean is also NaN or zero)
    time_diff_stats['time_diff_ratio'] = (
            time_diff_stats['std_time_diff'] / time_diff_stats['mean_time_diff']
    ).fillna(0)

    # 5-6. Unique merchants and ratio
    merchant_stats = df.groupby('card_id')['merchant_id'].nunique().rename('unique_merchants')
    merchant_ratio = (n_txn / merchant_stats).rename('txn_per_merchant')

    # 7-8. Unique MCCs and ratio
    mcc_stats = df.groupby('card_id')['merchant_mcc'].nunique().rename('unique_mccs')
    mcc_ratio = (n_txn / mcc_stats).rename('txn_per_mcc')

    # # 9-10. Most popular transaction type and its ratio
    # most_common_tx_type = df.groupby('card_id')['transaction_type'].agg(
    #     lambda x: x.value_counts().index[0]
    # ).rename('most_common_tx_type')
    #
    # most_common_tx_type_ratio = df.groupby('card_id')['transaction_type'].apply(
    #     lambda x: x.value_counts().iloc[0] / len(x)
    # ).rename('most_common_tx_type_ratio')

    # 11-12. Mean & std of transaction_amount_kzt
    amount_stats = df.groupby('card_id')['transaction_amount_kzt'].agg([
        ('mean_txn_amt', 'mean'),
        ('std_txn_amt', 'std')
    ])

    # 13. ECOM ratio
    ecom_ratio = df.groupby('card_id').apply(
        lambda x: (x['transaction_type'] == 'ECOM').sum() / len(x)
    ).rename('ecom_txn_ratio')

    # 14. Contactless ratio
    contactless_ratio = df.groupby('card_id').apply(
        lambda x: (x['pos_entry_mode'] == 'Contactless').sum() / len(x)
    ).rename('contactless_ratio')

    # 15. Total transaction amount
    total_amount = df.groupby('card_id')['transaction_amount_kzt'].sum().rename('total_txn_amt')

    # 16. Foreign spend ratio
    foreign_amt = df[df['acquirer_country_iso'] != 'KAZ'].groupby('card_id')['transaction_amount_kzt'].sum()
    foreign_ratio = (foreign_amt / total_amount).fillna(0).rename('foreign_spend_ratio')

    # Combine all features
    features = pd.concat([
        n_txn,
        time_diff_stats,
        merchant_stats,
        merchant_ratio,
        mcc_stats,
        mcc_ratio,
        # most_common_tx_type,
        # most_common_tx_type_ratio,
        amount_stats,
        ecom_ratio,
        contactless_ratio,
        total_amount,
        foreign_ratio
    ], axis=1)

    # Encode most_common_tx_type to integers
    # tx_type_mapping = {tx: idx for idx, tx in enumerate(features['most_common_tx_type'].dropna().unique(), start=1)}
    # features['most_common_tx_type_encoded'] = features['most_common_tx_type'].map(tx_type_mapping)

    features.index.name = 'card_id'
    return features


if __name__ == '__main__':
    from data_loader import load_transaction_data
    transaction_data = load_transaction_data('data/transactions.parquet')
    if transaction_data is not None:
        features = create_behavioral_features(transaction_data)
        print(features.head())
        print(features.shape)
        nan_locations = features[features.isna().any(axis=1)]
        for row_idx, row in nan_locations.iterrows():
            missing_cols = row[row.isna()].index.tolist()
            print(f"Row {row_idx} has NaNs in columns: {missing_cols}")
        print(features.loc[10453])

