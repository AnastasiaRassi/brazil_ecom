import logging, os, yaml
from dotenv import load_dotenv
load_dotenv()  
base_path = os.getenv("BASE_PATH")

def setup_logger(config_path = os.path.join(base_path,"config.yaml")):
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
    logger = setup_logger()
    logger.info("Logger is set up and ready to use.")
    
    return logging.getLogger(__name__)
