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

    df.drop_duplicates(inplace=True)


if __name__ == '__main__':
    from data_loader import load_transaction_data
    transaction_data = load_transaction_data('data/transactions.parquet')
    if transaction_data is not None:
        features = create_behavioral_features(transaction_data)
        print(features.head())