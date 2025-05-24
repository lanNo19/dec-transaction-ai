# main.py
from data_loader import load_transaction_data
from feature_engineering import create_behavioral_features
import pandas as pd

DATA_FILE_PATH = 'data/transactions.parquet'
LOW_MCC_THRESHOLD = 3

if __name__ == '__main__':
    transaction_data = load_transaction_data(DATA_FILE_PATH)
    if transaction_data is not None:
        behavioral_features = create_behavioral_features(transaction_data)
        print("\nFirst few rows of behavioral features:")
        print(behavioral_features.head())

        low_mcc_users = behavioral_features[behavioral_features['unique_mcc'] <= LOW_MCC_THRESHOLD]
        print(f"\nNumber of users with {LOW_MCC_THRESHOLD} or fewer unique MCCs: {len(low_mcc_users)}")
        if not low_mcc_users.empty:
            print("\nSome examples of users with low unique MCCs:")
            print(low_mcc_users.head())