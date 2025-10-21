import pandas as pd
from typing import List, Optional

# DataLoader class to load and validate CSV files

class DataLoader:
    # Initialize with optional required fields ( this is the combination of all of str cols, num cols and date cols found in config.yaml)
    def __init__(self, required_fields: Optional[List[str]] = None):
        self.required_fields = required_fields

    # Method to load CSV and validate required fields
    def load_csv(self, file_path: str, min_rows: int) -> pd.DataFrame:
        """
        Load a CSV file and validate required fields and minimum rows.
        args:
            file_path (str): Path to the CSV file.
            min_rows (int): Minimum number of rows required in the dataset. (within config.yaml)
        returns:
            pd.DataFrame: Loaded DataFrame if valid.
        """

        # Check if the file has a .csv extension
        if not file_path.lower().endswith('.csv'):
            raise ValueError("File must be a CSV")

        # Try reading the CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV: {e}")

        # Check for required columns
        if self.required_fields:
            # Identify missing fields
            missing_fields = [field for field in self.required_fields if field not in df.columns]
            # Raise error if any required fields are missing
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
        # Check for minimum number of rows
        if df.shape[0] < min_rows :
            raise ValueError("Data must have at least {min_rows} rows, please populate your dataset for effective training.")

        return df
        
    


