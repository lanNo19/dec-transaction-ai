# data_loader.py
import pandas as pd
from typing import Optional
from icontract import require, ensure

@require(lambda file_path: isinstance(file_path, str) and len(file_path) > 0)
@ensure(lambda result: result is None or isinstance(result, pd.DataFrame))
def load_transaction_data(file_path: str) -> Optional[pd.DataFrame]:
    """Loads transaction data from a Parquet file.

    :param file_path: The path to the Parquet file.
    :return: The loaded transaction data, or None if loading fails.
    """
    try:
        df = pd.read_parquet(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

if __name__ == '__main__':
    data = load_transaction_data('transactions.parquet')
    if data is not None:
        print(data.head())
        print(data.info())