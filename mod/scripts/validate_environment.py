#!/usr/bin/env python3
"""Environment validation script"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.logging_utils import setup_logging
import torch

def main():
    logger = setup_logging()
    logger.info("Validating environment...")
    
    # Check PyTorch and CUDA
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    logger.info("Environment validation complete!")

if __name__ == "__main__":
    main()
