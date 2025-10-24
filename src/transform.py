import pandas as pd, os, yaml, numpy as np, joblib
from typing import List, Optional
from src import GeneralUtils, Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from category_encoders import TargetEncoder

logger = GeneralUtils.setup_logger()
base_path = os.getenv("BASE_PATH")

# Load configuration file
config_path = os.path.join(base_path, 'config.yaml')
with open(config_path, "r") as f:
    config = yaml.safe_load(f) or {}

ARTIFACT_DIR = os.path.join(base_path, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


class DataTransformer:
    def __init__(self, production=False):
        dataset_path = os.path.join(base_path, 'data', 'preprocessed_dataset.csv')
        self.data = pd.read_csv(dataset_path)
        self.splitter = Splitter(self.data)
        self.production = production

        target_engineer_path = os.path.join(ARTIFACT_DIR, "target_engineer.joblib")
        if os.path.exists(target_engineer_path):
            self.target_engineer = joblib.load(target_engineer_path)
        else:
            self.target_engineer = TargetEngineer()

    def transform_data_pipeline(self):
        self.X, self.y = self.splitter.split_X_y()

        if not self.production:
            # training/validation mode
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.splitter.split_train_test_val()
            self.y_train = self.target_engineer.fit_transform(self.y_train)
            self.y_val = self.target_engineer.transform(self.y_val)
            self.y_test = self.target_engineer.transform(self.y_test)
            self.X_train, self.X_val, self.X_test = Transformer.full_transform(self.X_train, self.X_val, self.X_test, self.y_train)
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

        else:
            # production mode — load saved encoders and scalers
            self.y = self.target_engineer.transform(self.y)
            self.X = Transformer.single_transform(self.X)
            return self.X


class Splitter:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def split_X_y(self):
        y_name = config["data"]["target_unmodified"]
        if not y_name:
            raise ValueError(f"Invalid target name: {y_name}")
        self.X = self.data.drop(columns=[y_name])
        self.y = self.data[y_name]
        logger.info("X and y split done.")
        return self.X, self.y

    def split_train_test_val(self):
        test_size = config["data"]["test_size"]
        test_to_val = config["data"]["test_to_val"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_to_val, random_state=42, stratify=y_temp
        )
        logger.info("Train, test, and val splits done.")
        return X_train, X_val, X_test, y_train, y_val, y_test


class TargetEngineer:
    """Bins continuous target variable for classification tasks using training data as reference."""
    def __init__(self):
        self.bin_method = config["data"]["target_bin_method"]
        self.num_bins = config["data"].get("target_quantiles", 4)
        self.bins = None
        self.labels = None

    def fit(self, train_target: pd.Series):
        if self.bin_method == "equal-width":
            self.bins = [0, 5, 10, 15, 20, 25, 33]
            self.labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-33']
        elif self.bin_method == "quantile":
            self.bins = pd.qcut(train_target, q=self.num_bins, retbins=True, duplicates="drop")[1]
            self.labels = [f'Q{i+1}' for i in range(len(self.bins) - 1)]
        else:
            raise ValueError(f"Unknown binning method '{self.bin_method}'")
        return self.transform(train_target)

    def transform(self, target: pd.Series):
        if self.bins is None or self.labels is None:
            raise ValueError("Bins not fitted yet. Call fit() first.")
        return pd.cut(target, bins=self.bins, labels=self.labels, include_lowest=True)

    def fit_transform(self, train_target: pd.Series):
        self.fit(train_target)
        logger.info("Fit transform of target binning on train is done.")
        joblib.dump(self, os.path.join(ARTIFACT_DIR, "target_engineer.joblib"))
        return self.transform(train_target)

    def inverse_transform(self, binned_target: pd.Series):
        mapping = {label: (low, high) for label, low, high in zip(self.labels, self.bins[:-1], self.bins[1:])}
        return binned_target.map(mapping)


class Transformer:
    @staticmethod
    def full_transform(X_train, X_val, X_test, y_train):
        X_train, X_val, X_test = Transformer.target_encode(X_train, X_val, X_test, y_train)
        X_train, X_val, X_test = Transformer.scale(X_train, X_val, X_test)
        return X_train, X_val, X_test

    @staticmethod
    def single_transform(X):
        te = joblib.load(os.path.join(ARTIFACT_DIR, "target_encoder.joblib"))
        scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))

        X = X.copy()
        X["product_category_name_english"] = te.transform(X[["product_category_name_english"]])

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        X[numeric_cols] = scaler.transform(X[numeric_cols])

        logger.info("Single-batch production transform done.")
        return X

    @staticmethod
    def target_encode(X_train, X_val, X_test, y_train):
        te = TargetEncoder(cols=["product_category_name_english"], smoothing=0.3)
        te.fit(X_train[["product_category_name_english"]], y_train)

        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        for df in [X_train, X_val, X_test]:
            df["product_category_name_english"] = te.transform(df[["product_category_name_english"]])

        joblib.dump(te, os.path.join(ARTIFACT_DIR, "target_encoder.joblib"))
        logger.info("Target encoder fitted and saved.")
        return X_train, X_val, X_test

    @staticmethod
    def scale(X_train, X_val, X_test):
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        scaler = RobustScaler()

        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
        logger.info("RobustScaler fitted and saved.")
        return X_train, X_val, X_test
