import pandas as pd
from src import setup_logger

logger = setup_logger()

### ===== UTILS FOR PREPROCESSOR.PY =====
def string_handling(self, col, fix_names: dict = fix_names):
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



def replace_rare_categories(series: pd.Series, threshold: int, replacement: str = "other") -> pd.Series:
    counts = series.value_counts(dropna=True)
    rare = counts[counts < threshold].index
    result = series.where(~series.isin(rare), replacement)
    return result.astype('category')

def string_handling(series: pd.Series, fix_names: dict) -> pd.Series:
    series = series.str.strip().str.lower()
    series = series.replace(fix_names)
    return series.astype('category')

def failed_parses_handling(data: pd.DataFrame, col: str, parsed: pd.Series, dtype: str) -> pd.DataFrame:
    """ 
    Drops rows where parsing failed for the given dtype (datetime or numeric).
    Args: 
        data: 
        col (str): The name of the column to check. 
        parsed (pd.Series): The parsed series of the column. dtype 
        (str): The expected data type ('datetime' or 'numeric').
    Returns: 
        None, modifies the data attribute in place.
    """

    errors_mask = parsed.isna() & data[col].notna()
    if errors_mask.sum() > 0:
        logger.warning(f"Incorrectly formatted {dtype} values dropped: {errors_mask.sum()} rows in {col}.")
        logger.info(f"Shape before {data.shape[0]}")
        data = data.drop(index=data[errors_mask].index)
        logger.info(f"Shape after {data.shape[0]}")
    data[col] = parsed
    return data
    
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

def replace_rare_categories(self, series: pd.Series, threshold: int, replacement: str) -> pd.Series:
    """
    Replace categories that appear fewer than `threshold` times with `replacement`.
    Keeps NaNs as-is. Preserves categorical dtype if input was categorical.
    """
    # count only non-null values for thresholding
    counts = series.value_counts(dropna=True)
    logger.debug(f"Category counts (for replacement decision): {counts.to_dict()}")

    rare = counts[counts < threshold].index
    if len(rare) == 0:
        logger.debug("No rare categories to replace.")
        return series

    result = series.where(~series.isin(rare), replacement)

    # cast the column as categorical
    return result.astype('category')


