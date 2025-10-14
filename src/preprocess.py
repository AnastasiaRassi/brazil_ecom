"""
validate data
preprocessing (nulls-handling, outlier-handling encoding, scaling) ...
"""
import pandas as pd

timestamp_checks = {
    "approval_before_purchase": lambda d: d['order_approved_at'] < d['order_purchase_timestamp'],
    "carrier_before_approval": lambda d: d['order_delivered_carrier_date'] < d['order_approved_at'],
    "delivery_before_carrier": lambda d: d['order_delivered_customer_date'] < d['order_delivered_carrier_date'],
    "delivery_before_purchase": lambda d: d['order_delivered_customer_date'] < d['order_purchase_timestamp'],
    "estimated_before_purchase": lambda d: d['order_estimated_delivery_date'] < d['order_purchase_timestamp'],
}

    
class preprocessor:
    def __init__(self, full_data: pd.DataFrame):
        self.data = full_data

    def validate_data(self, date_cols: list, str_cols: list, num_cols: list):
        """
        Validates the data for nulls, duplicates, correct datatypes.
        Args:
            date_cols (list): List of columns expected to be of datetime type.  (within config.yaml)
            str_cols (list): List of columns expected to be of string type.  (within config.yaml)
            num_cols (list): List of columns expected to be of numeric type.  (within config.yaml)
        Returns: 
            Exceptions wherever data is invalid for this project.
        """

        # Check for null values
        if self.data.isnull().sum().sum() > 0:
            print("Data contains null values")
        else:
            print("No null values found in the data.")

        # Check for duplicates
        if self.data.duplicated().sum() > 0:
            print("Data contains duplicate rows, they will be removed.")
            self.data = self.data.drop_duplicates()
        else:
            print("No duplicate rows found in the data.")
        
        # enforce correct datatypes
        for col in self.data.columns:
            try:
                invalid_date_rows = {}
                if col in date_cols:
                    # Attempt to parse the column as datetime
                    parsed = pd.to_datetime(self.data[col], errors='coerce')
                    # Identify rows where parsing failed
                    errors_mask = parsed.isna() & self.data[col].notna()
                    # Store the invalid rows in invalid_date_rows
                    invalid_date_rows[col] = self.data.loc[errors_mask]
                    # Replace the original column with the parsed datetime
                    self.data[col] = parsed
                    print("Invalid date rows will be dropped, there are {invalid_date_rows[col].shape[0]} rows in total in {col}.")
                    # Drop the invalid date rows from the data
                    self.data.drop(index=invalid_date_rows.index)

                elif col in str_cols:
                    # Check for rare categories in 'product_category_name_english'
                    # If the column is 'product_category_name_english', replace rare categories with 'Other'
                    if col == 'product_category_name_english':
                        threshold = 50
                        counts = self.data['product_category_name_english'].value_counts()
                        rare_categories = counts[counts < threshold].index
                        self.data['product_category_name_english'] = self.data['product_category_name_english'].replace(rare_categories, 'Other')
                    # Convert the column to string type
                    self.data[col] = self.data[col].astype(str)

                elif col in num_cols:
                `   self.data[col] = pd.to_numeric(self.data[col])

            
                print(f"Column {col} not consistent with its required format! {e}")

