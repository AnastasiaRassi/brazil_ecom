import pandas as pd, os, yaml, numpy as np
from src import DataLoader, GeneralUtils
import joblib
from flaml import AutoML
from sklearn.metrics import f1_score, classification_report


logger = GeneralUtils.setup_logger()

base_path = os.getenv("BASE_PATH")

# Load configuration file
config_path = os.path.join(base_path, 'config.yaml')
with open(config_path, "r") as f:
    config = yaml.safe_load(f) or {}

class Trainer:
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        self.automl = AutoML()
    
    def train(self, time_budget=300, metric="f1_macro"):
        settings = {
            "time_budget": time_budget,
            "metric": metric,
            "task": "classification",
            "log_file_name": "flaml_classification.log",
            "estimator_list": ["lgbm", "rf", "xgboost"],
        }
        self.automl.fit(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            **settings
        )

    def evaluate(self):
        y_pred_val = self.automl.predict(self.X_val)
        y_pred_test = self.automl.predict(self.X_test)

        print("Validation F1:", f1_score(self.y_val, y_pred_val, average='macro'))
        print("Test F1:", f1_score(self.y_test, y_pred_test, average='macro'))
        print("\nClassification Report (Test):")
        print(classification_report(self.y_test, y_pred_test))

    def save_model(self, path="best_flaml_model.pkl"):
        joblib.dump(self.automl.model, path)
