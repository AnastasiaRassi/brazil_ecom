import pandas as pd, os
from src import setup_logger, string_handling, replace_rare_categories, failed_parses_handling


logger = setup_logger()

base_path='C:/Users/aalrassi/Documents/anastasiawork/ML_and_DS/brazil_ecom/'

timestamp_checks = {
    "approval_before_purchase": lambda d: d['order_approved_at'] < d['order_purchase_timestamp'],
    "carrier_before_approval": lambda d: d['order_delivered_carrier_date'] < d['order_approved_at'],
    "delivery_before_carrier": lambda d: d['order_delivered_customer_date'] < d['order_delivered_carrier_date'],
    "delivery_before_purchase": lambda d: d['order_delivered_customer_date'] < d['order_purchase_timestamp'],
    "estimated_before_purchase": lambda d: d['order_estimated_delivery_date'] < d['order_purchase_timestamp'],
    "date_too_old": lambda d: d['order_purchase_timestamp'] < pd.Timestamp("2016-01-01"),
    "date_too_new": lambda d: d['order_purchase_timestamp'] > pd.Timestamp.now(),
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
    def __init__(self, full_data: pd.DataFrame, config: dict = None, 
                config_path: str = os.join.path(base_path,'config.yaml')):

        self.data = full_data
        # load config either from dict or yaml file
        if config is not None:
            self.config = config
        elif config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}
    
        else:
            self.config = {}

        # defaults
        self.per_column_thresholds = self.config.get("rare_thresholds", {})


    def validate_data(self, date_cols: list, str_cols: list, num_cols: list, threshold: int, 
                      fix_names: dict, timestamp_checks: dict, numeric_checks:dict ) -> pd.DataFrame:
        
        logger.info("=== Starting data validation ===")

        """
        Validates the data for nulls, duplicates, correct datatypes.
        Args:
            date_cols (list): List of columns expected to be of datetime type.  (within config.yaml)
            str_cols (list): List of columns expected to be of string type.  (within config.yaml)
            num_cols (list): List of columns expected to be of numeric type.  (within config.yaml)
            threshold (int): Threshold for rare categories in categorical columns. (within config.yaml)
        Returns: 
                None, is a prerequisite to cleaning.
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
    
    def clean(self):
        self.validate_data(self.data)
        """
        once data has been validated, it gets cleaned here.
        """
        
    def outlier_handling(self, col: str):
        """
        Handles outliers in the specified column.
        Args:
            col (str): The name of the column to handle outliers in.
        returns:
            None, modifies the data attribute in place."""
        
        if self.data[col].dtype in ['object', 'category']:
            logger.info(f"String handling in string column: {col}, before outlier handling")
            self.string_handling(col)

        elif pd.api.types.is_numeric_dtype(self.data[col]):
            logger.info(f"Numeric outlier handling in numeric column: {col}")
            # Implement numeric outlier handling if needed

        # Handle rare categories in categorical column product_category_name_english
        if col == 'product_category_name_english':
            logger.info(f"Handling outliers in column: {col}")
            # Define a threshold for rare categories
            threshold = self.config.get("rare_thresholds", {})
            self.data[col] = self.replace_rare_categories(self.data[col], threshold)


        
        return self.data
    