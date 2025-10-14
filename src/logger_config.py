import logging
import yaml

def setup_logger(config_path="C:/Users/aalrassi/Documents/anastasiawork/ML_and_DS/brazil_ecom/config.yaml"):
    # Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    log_file = config["logger"]["log_file"]
    log_level = config["logger"].get("level", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # writes logs to file
            logging.StreamHandler()         # prints logs to console
        ]
    )

    return logging.getLogger(__name__)
