import pandas as pd
from src import setup_logger

logger = setup_logger()

### ===== UTILS FOR PREPROCESSORING STAGE =====

def string_handling(series: pd.Series, fix_names: dict) -> pd.Series:
    """
    Fixes the strings in the specified column by stripping whitespace, converting to lowercase,
    and replacing specified strings with their corrected versions. 
    Args:
        col (str): The name of the column to fix.
    Returns:
        None, modifies the data attribute in place.
    """ 
    logger.info(f"Fixing strings in column: {col}")
    sseries = series.str.strip().str.lower()
    series = series.replace(fix_names)
    return series.astype('category')

def replace_rare_categories(series: pd.Series, threshold: int, replacement: str) -> pd.Series:
    """ Replace categories that appear fewer than threshold times with replacement. 
    Keeps NaNs as-is. Preserves categorical dtype if input was categorical. 
    """
    counts = series.value_counts(dropna=True)
    rare = counts[counts < threshold].index
    logger.debug(f"Replacing rare categories with {str}")
    logger.info(f"shape before:{len(series)}")   
    result = series.where(~series.isin(rare), replacement)
    # cast the column as categorical
    logger.info(f"shape after:{len(series)}")
    return result.astype('category')

    

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
    
