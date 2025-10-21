import pandas as pd, os, yaml, holidays, numpy as np
from src import GeoCoder, PreprocessUtils, setup_logger

logger = setup_logger()

base_path='C:/Users/aalrassi/Documents/anastasiawork/ML_and_DS/brazil_ecom/'

config_path = os.path.join(base_path, 'config.yaml')
                 
with open(config_path, "r") as f:
    config = yaml.safe_load(f) or {}


class Preprocessor:
    def __init__(self, full_data: pd.DataFrame):
        self.data = full_data 

    def preprocess_pipeline(self):
            """
            Executes the full preprocessing pipeline:
            1. Validates schema and datatypes
            2. Cleans data inconsistencies
            3. Handles nulls (imputation or drop)
            4. Detects and handles outliers

              Returns:
            pd.DataFrame: fully preprocessed dataset
        """ 
            Validator.validate_data(self.data)
            Cleaner(self.data).clean_data()  
            NullHandler.handle_nulls(self.data)  
            # what couldn't be imputed gets dropped
            self.data.dropna(inplace=True)
            OutlierHandler.handle_outliers(self.data)
            if config["settings"]["task_type"] != 'regression':
                TargetEngineer.target_engineering(self.data)
            FeatureEngineer.feature_engineering(self.data)
            return self.data
    
class Validator:
    @staticmethod
    def validate_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the data for nulls, duplicates, correct datatypes.
        Returns: 
                None, is a prerequisite to cleaning.
        """

        logger.info("=== Starting data validation ===")
        
        # enforce correct datatypes
        for col in data.columns:
            date_cols = config["data"]["date_cols"]
            str_cols = config["data"]["str_cols"]
            num_cols = config["data"]["num_cols"]

            try:
                if col in date_cols:
                    # Attempt to parse the column as datetime
                    parsed = pd.to_datetime(data[col], errors='coerce')
                    # Handle rows where parsing failed
                    data = PreprocessUtils.failed_parses_handling(data, col , parsed, dtype='datetime')

                elif col in str_cols:
                    # data[col] = data[col].astype(str) was the old version, it will transform NAN to nan very bad
                    data[col] = data[col].astype("string")

                elif col in num_cols:
                    # Attempt to parse the column as numeric
                    parsed = pd.to_numeric(data[col], errors='coerce')
                    # Handle rows where parsing failed
                    data = PreprocessUtils.failed_parses_handling(data, col, parsed, dtype='numeric')

            except Exception as e:
                logger.error(f"Column {col} not consistent with its required format! {e}", exc_info=True)
        
        logger.info("=== Data validation complete ===")

class Cleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    # === MAIN ENTRY POINT ===
    def clean_data(self):
        logger.info("=== Starting data cleaning ===")
        self._filter_delivered_orders()
        self._remove_duplicates()
        self._clean_string_columns()
        self._check_timestamp_rules()
        self._check_numeric_rules()
        logger.info("=== Data cleaning complete ===")
        return self.data

    # === HELPERS ===

    def _filter_delivered_orders(self):
        if 'order_status' in self.data.columns:
            logger.info("Filtering delivered orders...")
            self.data = self.data[self.data['order_status'] == 'delivered'].copy()
            self.data.drop(columns=['order_status'], inplace=True)

    def _remove_duplicates(self):
        n_dupes = self.data.duplicated().sum()
        if n_dupes > 0:
            logger.warning(f"Data contains {n_dupes} duplicate rows — removing them.")
            before = self.data.shape[0]
            self.data = self.data.drop_duplicates()
            logger.info(f"Removed {before - self.data.shape[0]} duplicates.")
        else:
            logger.info("No duplicate rows found in the data.")

    def _clean_string_columns(self):
        fix_names = {
            'home_confort': 'home_comfort',
            'fashio_female_clothing': 'fashion_female_clothing',
            'costruction_tools_garden': 'construction_tools_garden',
            'costruction_tools_tools': 'construction_tools_tools'}
        
        for col in self.data.select_dtypes(include=['object', 'category']).columns:
            logger.info(f"Cleaning string column: {col}")
            self.data[col] = PreprocessUtils.string_handling(self.data[col], fix_names)

    def _check_timestamp_rules(self):
        timestamp_checks = {
            "order_purchase_timestamp": [
                {"name": "approval_before_purchase",
                 "requires": ["order_approved_at", "order_purchase_timestamp"],
                 "rule": lambda df: df["order_approved_at"] < df["order_purchase_timestamp"]},
                {"name": "estimated_before_purchase",
                 "requires": ["order_estimated_delivery_date", "order_purchase_timestamp"],
                 "rule": lambda df: df["order_estimated_delivery_date"] < df["order_purchase_timestamp"]},
                {"name": "date_too_old",
                 "requires": ["order_purchase_timestamp"],
                 "rule": lambda df: df["order_purchase_timestamp"] < pd.Timestamp("2016-01-01")},
                {"name": "date_too_new",
                 "requires": ["order_purchase_timestamp"],
                 "rule": lambda df: df["order_purchase_timestamp"] > pd.Timestamp.now()},
            ],
            "order_approved_at": [
                {"name": "carrier_before_approval",
                 "requires": ["order_delivered_carrier_date", "order_approved_at"],
                 "rule": lambda df: df["order_delivered_carrier_date"] < df["order_approved_at"]},
            ],
            "order_delivered_carrier_date": [
                {"name": "delivery_before_carrier",
                 "requires": ["order_delivered_customer_date", "order_delivered_carrier_date"],
                 "rule": lambda df: df["order_delivered_customer_date"] < df["order_delivered_carrier_date"]},
            ]
        }
        date_cols = config['data']['str_cols']
        for col in self.date_cols:
            rules_for_col = timestamp_checks.get(col, [])
            if not rules_for_col:
                continue
            for rule_def in rules_for_col:
                rule_name = rule_def["name"]
                req_cols = rule_def.get("requires", [])
                if not all(c in self.data.columns for c in req_cols):
                    logger.debug(f"Skipping rule {rule_name} for {col}: missing columns.")
                    continue

                try:
                    mask = rule_def["rule"](self.data)
                    if not isinstance(mask, pd.Series):
                        raise ValueError(f"Rule {rule_name} did not return a pd.Series")

                    mask = mask.fillna(False).astype(bool)
                    n_invalid = mask.sum()
                    if n_invalid > 0:
                        logger.warning(f"Dropping {n_invalid} invalid rows by rule {rule_name} on {col}")
                        self.data = self.data[~mask]
                except Exception as e:
                    logger.exception(f"Error while applying timestamp rule {rule_name} for {col}: {e}")

    def _check_numeric_rules(self):
        numeric_checks = {
            "negative_weight": lambda d: d['product_weight_g'] <= 0,
            "negative_length": lambda d: d['product_length_cm'] <= 0,
            "negative_height": lambda d: d['product_height_cm'] <= 0,
            "negative_width": lambda d: d['product_width_cm'] <= 0, }
        self.num_cols = config['data']['num_cols']
        for col in self.num_cols:
            if col not in self.data.columns:
                continue
            for name, rule in numeric_checks.items():
                mask = rule(self.data)
                n_invalid = mask.sum()
                if n_invalid > 0:
                    logger.warning(f"Dropping {n_invalid} logically invalid numeric rows in {col} ({name})")
                    self.data = self.data[~mask]
     
class OutlierHandler:
    @staticmethod
    def handle_outliers(data: pd.DataFrame) -> pd.DataFrame:
        """
        Handles outliers in the dataset.
        Returns:
            pd.DataFrame: cleaned data with outliers handled.
        """
        logger.info("=== Started Outlier handling ===")

        for col in data.columns: 
            if col == 'product_category_name_english':
                logger.info(f"Handling outliers in column: {col}")
                threshold = config["rare_thresholds"]["product_names"]
                data[col] = PreprocessUtils.replace_rare_categories(
                    data[col], threshold
                )

            elif col == 'num_items':
                logger.info(f"Handling outliers in column: {col}")
                threshold = config["rare_thresholds"]["num_items"]
                before_max = data[col].max()
                data[col] = data[col].clip(upper=threshold)
                logger.info(f"Clipped num_items max from {before_max} to {data[col].max()}")

            elif col == 'days_till_arrival':
                logger.info(f"Handling outliers in column: {col}")
                threshold = config["rare_thresholds"]["days_till_arrival_ratio"]
                quantile_98 = data[col].quantile(threshold)
                before_shape = data.shape[0]
                data = data[data[col] <= quantile_98]
                logger.info(
                    f"Dropped {before_shape - data.shape[0]} rows with days_till_arrival above {threshold} quantile ({quantile_98})"
                )

        logger.info("=== Outlier handling complete ===")
        return data

class NullHandler:
    @staticmethod
    def handle_nulls(data: pd.DataFrame) -> pd.DataFrame:

        geocoder = GeoCoder()
        logger.info("=== Started Nulls handling ===")
        NULL_TOKENS = {"nan", "none", "nat", "<na>", "null", ""}

        # Fill missing lat/lng
        data = PreprocessUtils.handle_null_zip_codes(data, geocoder, persist_cache=True)

        # Normalize null tokens
        for col in data.columns:
            s = data[col]
            mask = s.astype("string").str.lower().isin(NULL_TOKENS)
            if mask.any():
                data.loc[mask, col] = pd.NA

        logger.info("=== Nulls handling complete ===")
        return data


class TargetEngineer:
    @staticmethod
    def target_engineering(data: pd.DataFrame) -> pd.DataFrame:
        """
        Bins continuous target column (e.g., days_till_arrival) into discrete categories.
        Supports multiple binning strategies defined in config.
        """
        bin_method = config["data"]["target_bin_method"]

        if bin_method == "equal-width":
            # Fixed intervals
            bins = [0, 5, 10, 15, 20, 25, 33]
            labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-33']
            data['days_till_arrival_binned'] = pd.cut(
                data['days_till_arrival'], bins=bins, labels=labels, include_lowest=True
            )
            logger.info(f"Binned days_till_arrival into fixed intervals: {labels}")

        elif bin_method == "quantile":
            # Quantile-based bins
            num_bins = config["data"].get("target_quantiles", 4)
            labels = [f'Q{i+1}' for i in range(num_bins)]
            data['days_till_arrival_binned'] = pd.qcut(
                data['days_till_arrival'], q=num_bins, labels=labels
            )
            logger.info(f"Binned days_till_arrival into quantiles: {num_bins} classes")

        else:
            logger.debug(f"Unknown binning method '{bin_method}', skipping target binning.")

        return data

class FeatureEngineer:
    @staticmethod
    def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
        """
        Combines multiple feature engineering steps into a clear flow.
        Adds distance, processing times, cyclical features, speed features, traffic proxies, and flags.
        """
        perishable = {'food', 'drinks', 'food_drink','other','signaling_and_security','art'}
        slow_cats = {'christmas_supplies', 'garden_tools', 'home_comfort', 'dvds_blu_ray', 'construction_tools_tools'}
        fast_cats = {'drinks', 'food', 'construction_tools_lights'}

        # Distances and times
        data['distance_km'] = PreprocessUtils.haversine(
            data['customer_lat'], data['customer_lng'],
            data['seller_lat'], data['seller_lng']
        )
        data['processing_time_hours'] = (data['order_approved_at'] - data['order_purchase_timestamp']).dt.total_seconds() / 3600
        data['carrier_delay_hours'] = (data['order_delivered_carrier_date'] - data['order_approved_at']).dt.total_seconds() / 3600

        # Purchase & carrier date features
        for prefix, col in [('purchase', 'order_purchase_timestamp'), ('carrier_reception', 'order_delivered_carrier_date')]:
            data[f'{prefix}_dayofweek'] = data[col].dt.dayofweek
            data[f'{prefix}_year'] = data[col].dt.year
            data[f'{prefix}_month'] = data[col].dt.month
            data[f'{prefix}_day'] = data[col].dt.day
            data[f'{prefix}_hour'] = data[col].dt.hour
        data['carrier_reception_weekend'] = data['carrier_reception_dayofweek'].isin([5,6]).astype(int)

        # Holidays
        br_holidays = holidays.Brazil(years=[2016, 2017, 2018])
        data['is_holiday'] = data['order_delivered_carrier_date'].dt.date.apply(lambda x: int(x in br_holidays))

        # Product features
        data['product_volume_cm3'] = data['product_length_cm'] * data['product_width_cm'] * data['product_height_cm']
        data['product_density_g_per_cm3'] = data['product_weight_g'] / data['product_volume_cm3'].replace(0, np.nan)
        data['freight_ratio'] = data['freight_value'] / data['raw_price'].replace(0, np.nan)
        data['total_weight'] = data['product_weight_g'] * data['num_items']
        data['total_volume'] = data['product_volume_cm3'] * data['num_items']
        data['raw_price_per_item'] = data['raw_price'] / data['num_items'].replace(0, np.nan)
        data['installment_value'] = data['payment_value'] / data['payment_installments'].replace(0, np.nan)

        # Combined features
        data['weight_times_distance'] = data['total_weight'] * data['distance_km']
        data['processing_over_distance'] = data['processing_time_hours'] / data['distance_km'].replace(0, np.nan)
        data['is_perishable'] = data['product_category_name_english'].isin(perishable).astype(int)
        data['is_slow'] = data['product_category_name_english'].isin(slow_cats).astype(int)
        data['is_fast'] = data['product_category_name_english'].isin(fast_cats).astype(int)

        # Cyclical encoding
        cyclical_cols = {
            'purchase_month': 12, 'purchase_day': 31, 'purchase_hour': 24,
            'carrier_reception_month': 12, 'carrier_reception_day': 31, 'carrier_reception_hour': 24,
        }
        for col, max_val in cyclical_cols.items():
            data = PreprocessUtils.cyclical_encode(data, col, max_val)

        # Speed features
        data['observed_speed'] = data['distance_km'] / data['days_till_arrival']
        for group in ['product_category_name_english', 'seller_id', 'customer_id', 'customer_city', 'customer_state']:
            data[f'avg_speed_{group.split("_")[-1]}'] = PreprocessUtils.historical_avg_speed(data, group)
        # Fill missing avg speeds
        global_mean_speed = data['observed_speed'].mean()
        for col in [c for c in data.columns if c.startswith('avg_speed')]:
            data[col].fillna(global_mean_speed, inplace=True)
        # Predicted days
        for prefix in ['category', 'seller', 'customer', 'customer_city', 'customer_state', 'seller_city']:
            col_name = f'avg_speed_{prefix}'
            data[f'pred_days_{prefix}'] = data['distance_km'] / data[col_name]

        # Traffic proxies
        traffic_proxy = data.groupby(
            ['customer_city', 'customer_state', 'carrier_reception_dayofweek', 'carrier_reception_hour']
        ).size().reset_index(name='delivery_count')
        data = data.merge(
            traffic_proxy,
            on=['customer_city', 'customer_state', 'carrier_reception_dayofweek', 'carrier_reception_hour'],
            how='left'
        )
        data["traffic_smoothed"] = data.groupby(["customer_city", "customer_state"])["delivery_count"].transform(lambda x: x.rolling(3, min_periods=1, center=True).mean())
        data["traffic_ewm"] = data.groupby(["customer_city", "customer_state"])["delivery_count"].transform(lambda x: x.ewm(span=5, adjust=False).mean())

        # Drop unnecessary columns
        data.drop(columns=config["data"]["cols_to_drop"], inplace=True, errors='ignore')

        return data
