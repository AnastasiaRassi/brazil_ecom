"""
validate data
preprocessing (nulls-handling, outlier-handling encoding, scaling) ...
"""
import pandas as pd
from src import setup_logger
logger = setup_logger()

timestamp_checks = {
    "approval_before_purchase": lambda d: d['order_approved_at'] < d['order_purchase_timestamp'],
    "carrier_before_approval": lambda d: d['order_delivered_carrier_date'] < d['order_approved_at'],
    "delivery_before_carrier": lambda d: d['order_delivered_customer_date'] < d['order_delivered_carrier_date'],
    "delivery_before_purchase": lambda d: d['order_delivered_customer_date'] < d['order_purchase_timestamp'],
    "estimated_before_purchase": lambda d: d['order_estimated_delivery_date'] < d['order_purchase_timestamp'],
}

fix_names = {
    'home_confort': 'home_comfort',
    'fashio_female_clothing': 'fashion_female_clothing',
    'costruction_tools_garden': 'construction_tools_garden',
    'costruction_tools_tools': 'construction_tools_tools'
}

numeric_checks = {
    "negative_weight": lambda d: d['product_weight_g'] <= 0,
    "negative_length": lambda d: d['product_length_cm'] <= 0,
    "negative_height": lambda d: d['product_height_cm'] <= 0,
    "negative_width": lambda d: d['product_width_cm'] <= 0,
}


class Preprocessor:
    def __init__(self, full_data: pd.DataFrame):
        self.data = full_data

    def validate_data(self, date_cols: list, str_cols: list, num_cols: list):
        
        logger.info("=== Starting data validation ===")

        """
        Validates the data for nulls, duplicates, correct datatypes.
        Args:
            date_cols (list): List of columns expected to be of datetime type.  (within config.yaml)
            str_cols (list): List of columns expected to be of string type.  (within config.yaml)
            num_cols (list): List of columns expected to be of numeric type.  (within config.yaml)
        Returns: 
            Exceptions wherever data is invalid for this project, returns dataframe cleaned of invalid or illogical data.
        """
        
        logger.info("looking for nulls, duplicates, and correct datatypes in the data...")

        # Check for null values
        if self.data.isnull().sum().sum() > 0:
            logger.warning("Null values found in the data.")
        else:
            logger.info("No null values found in the data.")
        # Check for duplicates
        if self.data.duplicated().sum() > 0:

            logger.warning("Data contains duplicate rows, they will be removed.")

            logger.info(f"Dataset shape before dropping duplicates: {self.data.shape}")
            self.data = self.data.drop_duplicates()
            logger.info(f"Dataset shape after  dropping duplicates: {self.data.shape}")

        else:
            logger.info("No duplicate rows found in the data.")
        
        # enforce correct datatypes
        for col in self.data.columns:
            try:
                if col in date_cols:
                    # Attempt to parse the column as datetime
                    parsed = pd.to_datetime(self.data[col], errors='coerce')
                    # Handle rows where parsing failed
                    self.failed_parses_handling(col, parsed, dtype='datetime')
                    # Check for logical inconsistencies in date columns`
                    for name, rule in timestamp_checks.items():
                        # compute mask to catch date illogical values
                        mask = rule(self.data)  
                        logger.warning(f"Logically invalid date rows will be dropped, {mask.sum()} rows in total in {col}.")
                        logger.info(f"Dataset shape before dropping logically invalid date values: {self.data.shape}")
                        self.data = self.data[~mask]   
                        logger.info(f"Dataset shape after dropping logically invalid date values: {self.data.shape}")


                elif col in str_cols:
                    self.outlier_handling(col)
                    self.data[col] = self.data[col].astype(str)

                elif col in num_cols:
                    # Attempt to parse the column as numeric
                    parsed = pd.to_numeric(self.data[col], errors='coerce')
                    # Handle rows where parsing failed
                    self.failed_parses_handling(col, parsed, dtype='numeric')
                    # Convert the column to numeric type
                    self.data[col] = pd.to_numeric(self.data[col])
                    # Check for logical inconsistencies in numeric columns
                    for name, rule in numeric_checks.items():
                        # compute mask to catch numeric illogical values
                        mask = rule(self.data)  
                        logger.warning(f"Logically invalid numeric rows will be dropped, there are {mask.sum()} rows in total in {col}.")
                        logger.info(f"Dataset shape before dropping logically invalid numeric values: {self.data.shape}")
                        self.data = self.data[~mask]   
                        logger.info(f"Dataset shape after dropping logically invalid numeric values: {self.data.shape}")

            except Exception as e:
                logger.error(f"Column {col} not consistent with its required format! {e}", exc_info=True)
        
        logger.info("=== Data validation complete ===")
        return self.data
    
    def failed_parses_handling(self, col: str, parsed: pd.Series, dtype: str):
        """
        Drops rows where parsing failed for the given dtype (datetime or numeric).
        Args:
            col (str): The name of the column to check.
            parsed (pd.Series): The parsed series of the column.
            dtype (str): The expected data type ('datetime' or 'numeric').
        Returns:
            None, modifies the data attribute in place.
        """
        errors_mask = parsed.isna() & self.data[col].notna()
        if errors_mask.sum() > 0:
            logger.warning(f"Incorrectly formatted {dtype} values dropped: {errors_mask.sum()} rows in {col}." )
            logger.info(f"Dataset before dropping: {self.data.shape}")
            self.data.drop(index=self.data[errors_mask].index, inplace=True)
            logger.info(f"Dataset after dropping: {self.data.shape}")
        self.data[col] = parsed

    def outlier_handling(self, col: str):
        """
        Handles outliers in the specified column.
        Args:
            col (str): The name of the column to handle outliers in.
        returns:
            None, modifies the data attribute in place."""
        
        if col == 'product_category_name_english':
            logger.info(f"Handling outliers in column: {col}")
            # Fix names in the column
            self.string_handling(col)
            # Define a threshold for rare categories
            threshold = 50
            counts = self.data[col].value_counts()
            logger.info(f"Categories below threshold of {threshold} will be replaced with 'Other'")
            logger.info(f"Counts of categories before categorizing 'Other': {counts}")
            # Identify rare categories and replace them with 'Other'
            rare_categories = counts[counts < threshold].index
            self.data[col] = self.data[col].replace(rare_categories, 'Other')
            counts = self.data[col].value_counts()
            logger.info(f"Counts of categories after categorizing 'Other': {counts}")

    def string_handling(self,col):
        """
        Fixes the strings in the specified column by stripping whitespace, converting to lowercase,
        and replacing specified strings with their corrected versions. 
        Args:
            col (str): The name of the column to fix.
        Returns:
            None, modifies the data attribute in place.
        """ 
        logger.info(f"Fixing strings in column: {col}")
        self.data[col] = self.data[col].str.strip().str.lower()
        self.data[col] = self.data[col].replace(fix_names)
        self.data[col] = self.data[col].astype('category')
