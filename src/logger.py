import sys
import logging
from datetime import datetime

# Configure logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"

# Create logs directory if it doesn't exist
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_filepath = os.path.join(log_dir, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("WaterSensorLogger")

if __name__ == "__main__":
    logger.info("Logging has started")