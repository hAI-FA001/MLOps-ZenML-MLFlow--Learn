import logging

import pandas as pd

from src.data_cleaning import DataCleaning, DataPreProcessingStrategy


def get_data_for_test():
    try:
        df = pd.read_csv("./data/olist_merged.csv", nrows=30)
        df = df.sample(n=10)

        preprocess_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        
        df = df.drop(["review_score"], axis=1)
        result = df.to_json(orient="split")

        return result
    except Exception as e:
        logging.error(f"Error in getting test data: {e}")
        raise e