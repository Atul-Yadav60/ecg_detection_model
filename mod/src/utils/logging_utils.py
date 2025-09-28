"""
Logging utilities for ECG detection project
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_file=None, log_dir="experiments/logs"):
    """Setup logging configuration"""
    
    # Create log directory
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / log_file
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    
    return logging.getLogger(__name__)

def get_logger(name):
    """Get logger instance"""
    return logging.getLogger(name)
