#!/usr/bin/env python3
"""Model training script"""

import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.models.architectures.cnn_lstm import create_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training config path')
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(log_file="training.log")
    config = load_config(args.config)
    
    logger.info("Starting training...")
    logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Create model
    model = create_model(config.get('model', {}))
    logger.info(f"Created model: {model.__class__.__name__}")
    
    # Training would be implemented here
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
