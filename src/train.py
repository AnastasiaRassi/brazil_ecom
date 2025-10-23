import pandas as pd, os, yaml, numpy as np
from src import setup_logger, Preprocessor

logger = setup_logger()

base_path = os.getenv("BASE_PATH")

# Load configuration file
config_path = os.path.join(base_path, 'config.yaml')
with open(config_path, "r") as f:
    config = yaml.safe_load(f) or {}

