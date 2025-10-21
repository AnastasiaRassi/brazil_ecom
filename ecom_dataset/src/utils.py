import pandas as pd
from src import setup_logger
from geopy.geocoders import Nominatim
import time

logger = setup_logger()

# the preprocessing utils, for simplicity, have been specified for each stage.

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
    logger.info(f"Fixing strings in column: {series}")
    series = series.str.strip().str.lower().replace(fix_names)
    series = series.replace(fix_names)
    return series.astype('category')

def replace_rare_categories(series: pd.Series, threshold: int, replacement) -> pd.Series:
    """ Replace categories that appear fewer than threshold times with replacement. 
    Keeps NaNs as-is. Preserves categorical dtype if input was categorical. 
    """
    counts = series.value_counts(dropna=True)
    rare = counts[counts < threshold].index
    logger.debug(f"Replacing rare categories with {replacement}")
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
 
 

def get_lat_lng(city_name: str, state_name: str, country: str):
    """
    Computes the latitude and longitude of a location using Nominatim.

    Args:
        city_name (str): Name of the city.
        state_name (str or None): Name of the state (optional).
        country (str): Name of the country.

    Returns:
        tuple: (latitude, longitude) if found, else (None, None).
    """
    if not city_name or not country:
        logger.warning(f"Missing city or country: city='{city_name}', country='{country}'")
        return None, None

    query = f"{city_name}, {state_name}, {country}" if state_name else f"{city_name}, {country}"
    geolocator = Nominatim(user_agent="my_app")

    try:
        location = geolocator.geocode(query)
        if location:
            logger.debug(f"Geocoded {query}: ({location.latitude}, {location.longitude})")
            return location.latitude, location.longitude
        else:
            logger.warning(f"Geocoding failed for {query}")
            return None, None
    except Exception as e:
        logger.error(f"Error geocoding {query}: {e}")
        return None, None


def handle_null_zip_codes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing latitude and longitude for customer and seller locations
    using city, state, and country information.

    Args:
        data (pd.DataFrame): DataFrame containing columns:
            - '{prefix}_city', '{prefix}_state', '{prefix}_lat', '{prefix}_lng', 'country'
            - prefix is either 'customer' or 'seller'

    Returns:
        pd.DataFrame: DataFrame with imputed lat/lng values. Rows with missing
                      city or country remain NaN but are logged.
    """
    for prefix in ['customer', 'seller']:
        loc_missing_rows = 0
        missing = data[data[f'{prefix}_lat'].isnull() | data[f'{prefix}_lng'].isnull()]
        logger.debug(f"Started null handling zip codes for {prefix}... Total missing: {len(missing)}")

        for idx, row in missing.iterrows():
            if pd.isnull(row['country']) or pd.isnull(row[f'{prefix}_city']):
                loc_missing_rows += 1
                logger.debug(f"Skipping row {idx} for {prefix}: missing city or country")
                continue

            lat, lng = get_lat_lng(row[f'{prefix}_city'], row.get(f'{prefix}_state'), row['country'])
            data.at[idx, f'{prefix}_lat'] = lat
            data.at[idx, f'{prefix}_lng'] = lng
            time.sleep(1)  # Respect Nominatim rate limits

        logger.warning(f"{loc_missing_rows} rows were missing essential location info for {prefix} "
                       "and could not be imputed.")
    logger.info("Null zip code handling complete.")
    
    return data