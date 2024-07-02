import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting data from data_path
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting data from data_path
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path, nrows=10)  ## Set nrows to avoid "Out of memory" for my VM 


@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from data_path
    
    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """

    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error("Error while ingesting data: {e}")
        raise e