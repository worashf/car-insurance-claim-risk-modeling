
import os
import logging
import  pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load car insurance  data from csv file.
    :param file_path:
    :return:
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    try:
        df = pd.read_csv(file_path,  sep="|", low_memory=False)
        logger.info(f"Loaded {len(df)} records from {file_path}.")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load and process news data: {e}")


def load_clean_data(file_path:str) -> pd.DataFrame:
        """
        Load cleaned data from csv file form data/clean.
        :param df:
        :return:
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        try:
            df = pd.read_csv(file_path,  sep=",", low_memory=False)
            logger.info(f"Loaded {len(df)} records from {file_path}.")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load and process cleaned data: {e}")





