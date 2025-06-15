import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory setup
RAW_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "clean"

# Ensure processed directory exists
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_processed_data(df: pd.DataFrame):
    """
    Save processed DataFrame to a pipe-separated (.txt) file.
    """
    if df.empty:
        logger.warning("No data to save.")
        return None

    try:
        # Output file path
        filename = PROCESSED_DATA_DIR / "car_insurance_processed.txt"

        # Save to file with pipe separator
        df.to_csv(filename, index=False, sep="|")
        logger.info(f"Successfully saved {len(df)} records to {filename}")
        return df

    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}", exc_info=True)
        return None
