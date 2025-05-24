# feature_engineering.py
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
    if df is None or df.empty:
        print("Input DataFrame is empty or None.")
        return pd.DataFrame()

    customer_features = df.groupby('card_id').agg(
        total_spend=('transaction_amount_kzt', 'sum'),
        number_of_transactions=('transaction_id', 'count'),
        average_transaction_amount=('transaction_amount_kzt', 'mean'),
        unique_merchants=('merchant_id', 'nunique'),
        unique_mcc=('merchant_mcc', 'nunique') # We already have this
    )

    # Calculate transactions per month (requires transaction_timestamp)
    if 'transaction_timestamp' in df.columns:
        df['transaction_month'] = pd.to_datetime(df['transaction_timestamp']).dt.to_period('M')
        monthly_transactions = df.groupby(['card_id', 'transaction_month']).size().reset_index(name='monthly_transaction_count')
        avg_monthly_transactions = monthly_transactions.groupby('card_id')['monthly_transaction_count'].mean().rename('average_monthly_transactions')
        customer_features = customer_features.join(avg_monthly_transactions, how='left')

        # Calculate recency
        latest_timestamp = pd.to_datetime(df['transaction_timestamp']).max()
        df['time_diff'] = (latest_timestamp - pd.to_datetime(df['transaction_timestamp'])).dt.days
        recency = df.groupby('card_id')['time_diff'].min().rename('recency_days')
        customer_features = customer_features.join(recency, how='left')
    else:
        print("Warning: 'transaction_timestamp' column not found, skipping time-based feature calculation.")

    print("Behavioral features created.")
    customer_features.index.name = 'card_id'
    return customer_features

if __name__ == '__main__':
    from data_loader import load_transaction_data
    transaction_data = load_transaction_data('data/transactions.parquet')
    if transaction_data is not None:
        features = create_behavioral_features(transaction_data)
        print(features.head())