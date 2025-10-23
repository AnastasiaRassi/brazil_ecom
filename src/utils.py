import pandas as pd, os, yaml, json, time, logging, numpy as np
from pathlib import Path
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from src import GeneralUtils

logger = GeneralUtils.setup_logger()
base_path = os.getenv("BASE_PATH")
config_path = os.path.join(base_path, 'config.yaml')

# Load config
with open(config_path, "r") as f:
    config = yaml.safe_load(f) or {}

import  os, yaml
load_dotenv()  

class GeoCoder:
    def __init__(self, user_agent=None, cache_path=None):

        geocoding_cfg = config.get("geocoding", {})
        user_agent = user_agent or geocoding_cfg.get("user_agent", "geo_pipeline")
        min_delay = geocoding_cfg.get("min_delay_seconds", 1)
        max_retries = geocoding_cfg.get("max_retries", 2)
        error_wait = geocoding_cfg.get("error_wait_seconds", 2.0)

        # Setup geolocator with rate limiter
        self._geolocator = Nominatim(user_agent=user_agent, timeout=10)
        self._geocode = RateLimiter(
            self._geolocator.geocode,
            min_delay_seconds=min_delay,
            max_retries=max_retries,
            error_wait_seconds=error_wait
        )

        # Cache path from config or default
        cache_path = cache_path or config.get(
            "paths", {}
        ).get("GEOCODE_CACHE_PATH", Path(base_path) / "geocode_cache.json")
        self.cache_path = Path(cache_path)

        # Load existing cache or start empty
        self.cache = self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as fh:
                    return json.load(fh) or {}
            except Exception as e:
                logger.warning(f"Failed to load geocode cache: {e}")
        return {}

    def geocode_one(self, query: str):
        if not query:
            return None, None
        if query in self.cache:
            return tuple(self.cache[query])
        try:
            loc = self._geocode(query)
            latlng = (loc.latitude, loc.longitude) if loc else (None, None)
            self.cache[query] = latlng
            return latlng
        except Exception as e:
            logger.error(f"Geocoding error for '{query}': {e}")
            self.cache[query] = (None, None)
            return None, None

    def save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump(self.cache, fh, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save geocode cache: {e}")



class PreprocessUtils:
    @staticmethod
    def string_handling(series: pd.Series, fix_names: dict) -> pd.Series:
        """Strip whitespace, lowercase, and replace strings based on mapping."""
        logger.info(f"Cleaning strings in column {series.name}")
        if fix_names:
            mapping = {k.lower(): v for k, v in fix_names.items()}
            series = series.replace(mapping)
        return series.astype("category")

    @staticmethod
    def replace_rare_categories(series: pd.Series, threshold: int, replacement='Other') -> pd.Series:
        """Replace rare categories with 'Other', preserves NaNs."""
        counts = series.value_counts(dropna=True)
        rare = counts[counts < threshold].index.tolist()
        logger.debug(f"Replacing {len(rare)} rare categories in {series.name}")
        return series.where(~series.isin(rare), replacement).astype('category')

    @staticmethod
    def failed_parses_handling(data: pd.DataFrame, col: str, parsed: pd.Series, dtype: str) -> pd.DataFrame:
        errors_mask = parsed.isna() & data[col].notna()
        n_errors = int(errors_mask.sum())
        if n_errors > 0:
            logger.warning(f"Dropped {n_errors} incorrectly formatted {dtype} rows in {col}")
            data = data.drop(index=data[errors_mask].index)
        data.loc[:, col] = parsed.reindex(index=data.index)
        return data

    @staticmethod
    def handle_null_zip_codes(data: pd.DataFrame, geocoder: GeoCoder, persist_cache: bool = True) -> pd.DataFrame:
        """Fill missing lat/lng using geocoder object."""
        for prefix in ["customer", "seller"]:
            lat_col, lng_col = f"{prefix}_lat", f"{prefix}_lng"
            city_col, state_col = f"{prefix}_city", f"{prefix}_state"
            if not {lat_col, lng_col, city_col, state_col, "country"}.issubset(data.columns):
                continue
            mask_need = (data[lat_col].isna() | data[lng_col].isna()) & data[city_col].notna() & data["country"].notna()
            combos = data.loc[mask_need, [city_col, state_col, "country"]].drop_duplicates().values.tolist()
            combo_to_latlng = {tuple(c): geocoder.geocode_one(f"{c[0]}, {c[1]}, {c[2]}") for c in combos}

            data[lat_col] = data.apply(lambda r: r[lat_col] if pd.notna(r[lat_col]) else combo_to_latlng.get((r[city_col], r[state_col], r["country"]))[0], axis=1)
            data[lng_col] = data.apply(lambda r: r[lng_col] if pd.notna(r[lng_col]) else combo_to_latlng.get((r[city_col], r[state_col], r["country"]))[1], axis=1)

        if persist_cache:
            geocoder.save_cache()
        logger.info("Null zip code handling complete.")
        return data

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Vectorized Haversine distance in km."""
        lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        return 6371 * c

    @staticmethod
    def cyclical_encode(df, col, max_val):
        df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        return df
    
    @staticmethod
    def historical_avg_speed(df, group_col: str, target_col='observed_speed'):
        df = df.sort_values('order_purchase_timestamp')
        return df.groupby(group_col)[target_col].transform(lambda s: s.shift().expanding().mean())
    

class GeneralUtils:
    @staticmethod
    def setup_logger(config_path=None):
        if config_path is None:
            config_path = os.path.join(base_path, "config.yaml")
        
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        log_file = config["logger"]["log_file"]
        log_level = config["logger"].get("level", "INFO").upper()

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info("Logger is set up and ready to use.")
        
        return logger
