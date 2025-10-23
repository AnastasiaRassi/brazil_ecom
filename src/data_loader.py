import pandas as pd, os, yaml, numpy as np
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

class DataLoader:
    def __init__(self, data: pd.DataFrame):
        self.preprocessor = Preprocessor(data)
        self.data = self.preprocessor.preprocess_pipeline()
        self.splitter = Splitter(self.data)
        self.target_engineer = TargetEngineer()
    
    def load_data(self):
        self.X, self.y = self.splitter.split_X_y()
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.splitter.split_train_test_val()
        self.y_train = self.target_engineer.fit_transform(self.y_train)
        self.y_val = self.target_engineer.transform(self.y_val)
        self.y_test = self.target_engineer.transform(self.y_test)
        self.X_train, self.X_val, self.X_test = Transformer.full_transform(self.X_train, self.X_val, self.X_test, self.y_train)
    
class Splitter:
    def __init__(self, data:pd.DataFrame):
        self.data = data
    
    def split_X_y(self):
        y_name = config["data"]["target_unmodified"]
        if y_name:
            self.X = self.data.drop(columns=[y_name])
            self.y = self.data[y_name]
            logger.info("X and y split done.")
        else:
            raise ValueError(f"y_name is {y_name} - invalid.")
        return self.X, self.y
    
    def split_train_test_val(self):         
        test_size = config["data"]["test_size"]
        test_to_val = config["data"]["test_to_val"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size = test_size, random_state=42, stratify = self.y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size = test_to_val, random_state=42, stratify = y_temp
        )
        logger.info("Train test and val splits done.")
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test


class TargetEngineer:
    """Bins continuous target variable for classification tasks using training data as reference."""
    def __init__(self):
        self.bin_method =  config["data"]["target_bin_method"]
        self.num_bins = config["data"].get("target_quantiles", 4)
        self.bins = None
        self.labels = None

    def fit(self, train_target: pd.Series):
        """Compute bin edges and labels from training data."""
        if self.bin_method == "equal-width":
            self.bins = [0, 5, 10, 15, 20, 25, 33]
            self.labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-33']
            logger.debug(f"Bin method used:{self.bin_method}")
        elif self.bin_method == "quantile":
            self.bins = pd.qcut(train_target, q=self.num_bins, retbins=True, duplicates="drop")[1]
            self.labels = [f'Q{i+1}' for i in range(len(self.bins)-1)]
            logger.debug(f"Bin method used:{self.bin_method}")
        else:
            raise ValueError(f"Unknown binning method '{self.bin_method}'")

        return self.transform(train_target)

    def transform(self, target: pd.Series):
        """Apply pre-computed bins/labels to any dataset."""
        if self.bins is None or self.labels is None:
            raise ValueError("Bins not fitted yet. Call fit() first.")
        return pd.cut(target, bins=self.bins, labels=self.labels, include_lowest=True)
    

    def fit_transform(self, train_target: pd.Series):
        """Fit bins and transform training data in one step."""
        self.fit(train_target)
        logger.info("Fit transform of target binning on train is done.")
        return self.transform(train_target)

class Transformer:
    @staticmethod
    def full_transform(X_train, X_val, X_test, y_train):
        X_train, X_val, X_test = Transformer.target_encode(X_train, X_val, X_test, y_train)
        X_train, X_val, X_test = Transformer.scale(X_train, X_val, X_test, y_train)


    @staticmethod
    def target_encode(X_train: pd.DataFrame,  X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series):
        te = TargetEncoder(cols=["product_category_name_english"], smoothing=0.3)
        te.fit(X_train[["product_category_name_english"]], y_train)

        X_train.loc[:, "product_category_name_english"] = te.transform(X_train[["product_category_name_english"]])
        X_val.loc[:, "product_category_name_english"] = te.transform(X_val[["product_category_name_english"]])
        X_test.loc[:, "product_category_name_english"] = te.transform(X_test[["product_category_name_english"]])

        return X_train, X_val, X_test
    
    @staticmethod
    def scale(X_train: pd.DataFrame,  X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series):
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        scaler = RobustScaler()
        X_train[numeric_cols] = pd.DataFrame(
            scaler.fit_transform(X_train[numeric_cols]),
            columns=numeric_cols,
            index=X_train.index
        )
        X_val[numeric_cols] = pd.DataFrame(
            scaler.transform(X_val[numeric_cols]),
            columns=numeric_cols,
            index=X_val.index
        )
        X_test[numeric_cols] = pd.DataFrame(
            scaler.transform(X_test[numeric_cols]),
            columns=numeric_cols,
            index=X_test.index
        )
        return X_train, X_val, X_test


