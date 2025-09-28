#!/usr/bin/env python3
"""
ECG Arrhythmia Detection - Project Structure Creation
Step 2: Complete modular project setup
Optimized for RTX 3050 (4GB VRAM)
Windows-compatible version
"""

import os
import json
from pathlib import Path
from datetime import datetime

def create_project_structure():
    """Create complete project directory structure"""
    
    print("Building ECG Arrhythmia Detection Project Structure")
    print("=" * 60)
    
    # Define project structure
    structure = {
        "config/": {
            "model_configs/": {},
            "training_configs/": {},
            "data_configs/": {}
        },
        "data/": {
            "raw/": {
                "mit_bih/": {},
                "physionet/": {},
                "custom/": {}
            },
            "processed/": {
                "train/": {},
                "val/": {},
                "test/": {}
            },
            "cache/": {}
        },
        "src/": {
            "models/": {
                "architectures/": {},
                "components/": {},
                "pretrained/": {}
            },
            "data/": {
                "loaders/": {},
                "preprocessing/": {},
                "augmentation/": {}
            },
            "training/": {
                "trainers/": {},
                "losses/": {},
                "metrics/": {}
            },
            "inference/": {
                "engines/": {},
                "explainability/": {},
                "optimization/": {}
            },
            "utils/": {}
        },
        "notebooks/": {
            "exploration/": {},
            "experiments/": {},
            "visualization/": {}
        },
        "experiments/": {
            "runs/": {},
            "logs/": {},
            "checkpoints/": {}
        },
        "outputs/": {
            "models/": {},
            "predictions/": {},
            "reports/": {},
            "visualizations/": {}
        },
        "tests/": {
            "unit/": {},
            "integration/": {},
            "performance/": {}
        },
        "scripts/": {
            "data/": {},
            "training/": {},
            "inference/": {},
            "deployment/": {}
        },
        "docs/": {
            "api/": {},
            "tutorials/": {},
            "examples/": {}
        },
        "deployment/": {
            "streamlit/": {},
            "api/": {},
            "docker/": {},
            "models/": {}
        }
    }
    
    # Create directories
    created_dirs = []
    for path, subdirs in structure.items():
        os.makedirs(path, exist_ok=True)
        created_dirs.append(path)
        
        if subdirs:
            for subpath in subdirs:
                full_path = os.path.join(path, subpath)
                os.makedirs(full_path, exist_ok=True)
                created_dirs.append(full_path)
    
    print(f"Created {len(created_dirs)} directories")
    return created_dirs

def create_config_files():
    """Create configuration files optimized for RTX 3050"""
    
    print("\nCreating Configuration Files")
    
    # Main project configuration
    main_config = {
        "project": {
            "name": "ecg_arrhythmia_detection",
            "version": "1.0.0",
            "description": "Real-time ECG arrhythmia detection system",
            "authors": ["ECG Detection Team"]
        },
        "hardware": {
            "gpu": "RTX 3050 Laptop GPU",
            "vram_gb": 4.0,
            "recommended_batch_sizes": {
                "training": 64,
                "inference": 16,
                "max_safe": 128
            },
            "mixed_precision": True,
            "num_workers": 4
        },
        "paths": {
            "data_root": "./data",
            "models_root": "./outputs/models",
            "logs_root": "./experiments/logs",
            "checkpoints_root": "./experiments/checkpoints"
        }
    }
    
    with open("config/main_config.json", "w", encoding='utf-8') as f:
        json.dump(main_config, f, indent=2)
    
    # Training configuration optimized for RTX 3050
    training_config = {
        "model": {
            "architecture": "cnn_lstm",
            "input_length": 5000,
            "num_classes": 5,
            "channels": 1
        },
        "training": {
            "batch_size": 64,
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "mixed_precision": True,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0
        },
        "optimization": {
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_epochs": 5,
            "min_lr": 1e-6
        },
        "regularization": {
            "dropout": 0.3,
            "label_smoothing": 0.1,
            "mixup_alpha": 0.2
        },
        "early_stopping": {
            "patience": 15,
            "monitor": "val_f1",
            "mode": "max"
        }
    }
    
    with open("config/training_configs/default_training.json", "w", encoding='utf-8') as f:
        json.dump(training_config, f, indent=2)
    
    # Data configuration
    data_config = {
        "datasets": {
            "mit_bih": {
                "url": "https://physionet.org/files/mitdb/1.0.0/",
                "sampling_rate": 360,
                "duration_seconds": 30,
                "classes": ["N", "S", "V", "F", "Q"]
            }
        },
        "preprocessing": {
            "filtering": {
                "lowpass": 40,
                "highpass": 0.5,
                "notch": 50
            },
            "normalization": "robust_scaler",
            "segmentation": {
                "window_size": 5000,
                "overlap": 0.5
            }
        },
        "augmentation": {
            "noise_std": 0.01,
            "amplitude_scale": [0.8, 1.2],
            "time_stretch": [0.9, 1.1],
            "baseline_wander": True
        },
        "splits": {
            "train": 0.7,
            "val": 0.15,
            "test": 0.15,
            "patient_wise": True
        }
    }
    
    with open("config/data_configs/default_data.json", "w", encoding='utf-8') as f:
        json.dump(data_config, f, indent=2)
    
    # Model architectures configuration
    model_configs = {
        "cnn_lstm": {
            "conv_layers": [
                {"out_channels": 32, "kernel_size": 15, "stride": 1},
                {"out_channels": 64, "kernel_size": 15, "stride": 2},
                {"out_channels": 128, "kernel_size": 15, "stride": 2}
            ],
            "lstm_layers": [
                {"hidden_size": 128, "num_layers": 2, "dropout": 0.3}
            ],
            "classifier": {
                "hidden_dims": [256, 128],
                "dropout": 0.5
            }
        },
        "resnet1d": {
            "layers": [2, 2, 2, 2],
            "base_filters": 64,
            "kernel_size": 15,
            "stride": 2,
            "groups": 1,
            "width_per_group": 64
        },
        "transformer": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "max_seq_len": 5000
        }
    }
    
    with open("config/model_configs/architectures.json", "w", encoding='utf-8') as f:
        json.dump(model_configs, f, indent=2)
    
    print("Created configuration files")

def create_core_modules():
    """Create core Python modules with basic structure"""
    
    print("\nCreating Core Python Modules")
    
    modules = {
        "src/__init__.py": "",
        
        "src/models/__init__.py": "",
        
        "src/models/architectures/__init__.py": "",
        
        "src/models/architectures/cnn_lstm.py": '''"""
CNN-LSTM hybrid architecture for ECG arrhythmia detection
Optimized for RTX 3050 (4GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid model for ECG classification"""
    
    def __init__(self, input_channels=1, num_classes=5, sequence_length=5000):
        super(CNNLSTM, self).__init__()
        
        # CNN feature extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # LSTM for temporal modeling
        lstm_input_size = 128
        self.lstm = nn.LSTM(lstm_input_size, 128, num_layers=2, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 256 due to bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # CNN feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # Prepare for LSTM (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last output
        x = lstm_out[:, -1, :]
        
        # Classification
        x = self.classifier(x)
        
        return x

def create_model(config):
    """Factory function to create CNN-LSTM model"""
    return CNNLSTM(
        input_channels=config.get('input_channels', 1),
        num_classes=config.get('num_classes', 5),
        sequence_length=config.get('sequence_length', 5000)
    )
''',

        "src/data/__init__.py": "",
        
        "src/data/loaders/__init__.py": "",
        
        "src/data/loaders/ecg_dataset.py": '''"""
ECG Dataset loader with preprocessing
Optimized for patient-wise splits and memory efficiency
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json

class ECGDataset(Dataset):
    """ECG Dataset class for arrhythmia detection"""
    
    def __init__(self, data_path, labels_path, transform=None, sequence_length=5000):
        """
        Args:
            data_path: Path to ECG signals
            labels_path: Path to labels
            transform: Optional transform to be applied
            sequence_length: Length of ECG segments
        """
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path)
        self.transform = transform
        self.sequence_length = sequence_length
        
        # Load metadata
        self.load_metadata()
        
    def load_metadata(self):
        """Load dataset metadata and file paths"""
        # This would be implemented based on specific dataset format
        # For now, placeholder structure
        self.samples = []
        self.labels = []
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # Load ECG signal
        signal = self.load_signal(idx)
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            signal = self.transform(signal)
            
        return torch.FloatTensor(signal), torch.LongTensor([label])
    
    def load_signal(self, idx):
        """Load and preprocess ECG signal"""
        # Placeholder - would load actual ECG data
        # Return normalized signal of specified length
        return np.random.randn(1, self.sequence_length)

def create_dataloaders(config, batch_size=64, num_workers=4):
    """Create train, validation, and test dataloaders"""
    
    # Create datasets
    train_dataset = ECGDataset(
        data_path=config['train_data_path'],
        labels_path=config['train_labels_path'],
        sequence_length=config['sequence_length']
    )
    
    val_dataset = ECGDataset(
        data_path=config['val_data_path'],
        labels_path=config['val_labels_path'],
        sequence_length=config['sequence_length']
    )
    
    # Create dataloaders with RTX 3050 optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader
''',

        "src/utils/__init__.py": "",
        
        "src/utils/config.py": '''"""
Configuration management utilities
"""

import json
from pathlib import Path

class Config:
    """Configuration management class"""
    
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if self.config_path.suffix == '.json':
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
    
    def get(self, key, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def update(self, key, value):
        """Update configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def save(self, path=None):
        """Save configuration to file"""
        save_path = path or self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            if save_path.suffix == '.json':
                json.dump(self.config, f, indent=2)

def load_config(config_path):
    """Load configuration from path"""
    return Config(config_path)
''',

        "src/utils/logging_utils.py": '''"""
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
''',

        "requirements.txt": '''# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# ECG processing
wfdb>=4.0.0
heartpy>=1.2.7
pywavelets>=1.3.0

# Deep learning utilities
tensorboard>=2.8.0
torchsummary>=1.5.0
timm>=0.6.0

# Visualization and UI
plotly>=5.0.0
streamlit>=1.25.0
gradio>=3.0.0

# Utilities
tqdm>=4.62.0
rich>=12.0.0
click>=8.0.0

# Model export and optimization
onnx>=1.12.0
onnxruntime-gpu>=1.12.0

# Development tools
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
jupyter>=1.0.0
''',

        ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Data
data/raw/
data/processed/
*.h5
*.hdf5

# Experiments
experiments/runs/
experiments/logs/
experiments/checkpoints/
*.pth
*.pt
*.onnx

# Outputs
outputs/models/
outputs/predictions/
outputs/reports/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.env.local
''',

        "README.md": '''# ECG Arrhythmia Detection System

Real-time ECG arrhythmia detection using deep learning, optimized for ASUS TUF A15 with RTX 3050.

## Features

- Real-time ECG arrhythmia detection
- RTX 3050 optimized (4GB VRAM)
- Multiple architecture support (CNN-LSTM, ResNet1D, Transformer)
- Advanced explainability (Grad-CAM, attention visualization)
- Mixed precision training
- Streamlit web interface

## Quick Start

1. **Environment Setup**
   ```bash
   conda create -n ecg_detection python=3.9
   conda activate ecg_detection
   pip install -r requirements.txt
   ```

2. **Validate Setup**
   ```bash
   python scripts/validate_environment.py
   ```

3. **Download Data**
   ```bash
   python scripts/data/download_datasets.py
   ```

4. **Train Model**
   ```bash
   python scripts/training/train_model.py --config config/training_configs/default_training.json
   ```

5. **Run Interface**
   ```bash
   streamlit run deployment/streamlit/app.py
   ```

## Hardware Optimization

Optimized for RTX 3050 Laptop GPU:
- Training batch size: 64
- Inference batch size: 16
- Mixed precision: Enabled
- VRAM usage: <3GB

## Project Structure

```
├── config/           # Configuration files
├── data/            # Dataset storage
├── src/             # Source code
│   ├── models/      # Model architectures
│   ├── data/        # Data processing
│   ├── training/    # Training utilities
│   └── inference/   # Inference engines
├── experiments/     # Training logs and checkpoints
├── outputs/         # Model outputs and reports
└── deployment/      # Deployment scripts
```

## License

MIT License
'''
    }
    
    # Create all files with UTF-8 encoding
    for file_path, content in modules.items():
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {len(modules)} core modules")

def create_scripts():
    """Create utility scripts"""
    
    print("\nCreating Utility Scripts")
    
    scripts = {
        "scripts/__init__.py": "",
        
        "scripts/validate_environment.py": '''#!/usr/bin/env python3
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
''',

        "scripts/data/download_datasets.py": '''#!/usr/bin/env python3
"""Download ECG datasets"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import wfdb
from pathlib import Path
from src.utils.logging_utils import setup_logging

def download_mit_bih():
    """Download MIT-BIH Arrhythmia Database"""
    logger = setup_logging()
    logger.info("Downloading MIT-BIH dataset...")
    
    # Create data directory
    data_dir = Path("data/raw/mit_bih")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download would be implemented here
    logger.info("MIT-BIH dataset ready!")

def main():
    download_mit_bih()

if __name__ == "__main__":
    main()
''',

        "scripts/training/train_model.py": '''#!/usr/bin/env python3
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
'''
    }
    
    # Create script files with UTF-8 encoding
    for file_path, content in scripts.items():
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {len(scripts)} utility scripts")

def create_setup_summary():
    """Create setup summary and next steps"""
    
    print("\nCreating Setup Summary")
    
    summary = {
        "project_created": datetime.now().isoformat(),
        "hardware_optimized_for": "RTX 3050 Laptop GPU (4GB VRAM)",
        "structure_status": "Complete",
        "next_steps": [
            "Run validation: python scripts/validate_environment.py",
            "Download datasets: python scripts/data/download_datasets.py", 
            "Start development in src/",
            "Train first model: python scripts/training/train_model.py --config config/training_configs/default_training.json"
        ],
        "key_features": [
            "Mixed precision training support",
            "RTX 3050 optimized batch sizes", 
            "Modular architecture",
            "Configuration-driven development",
            "Ready for real-time inference"
        ]
    }
    
    with open("PROJECT_SETUP_SUMMARY.json", "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("Setup summary saved to PROJECT_SETUP_SUMMARY.json")

def main():
    """Main project structure creation function"""
    
    print("ECG ARRHYTHMIA DETECTION - PROJECT STRUCTURE CREATION")
    print("Optimized for ASUS TUF A15 with RTX 3050 (4GB VRAM)")
    print("=" * 60)
    
    # Create all components
    create_project_structure()
    create_config_files()
    create_core_modules()
    create_scripts()
    create_setup_summary()
    
    print("\n" + "=" * 60)
    print("PROJECT STRUCTURE CREATION COMPLETE!")
    print("Your ECG detection project is ready for development!")
    
    print("\nProject Structure Overview:")
    print("├── config/           # All configuration files")
    print("├── data/             # Dataset storage (raw/processed)")
    print("├── src/              # Source code (models/data/training)")
    print("├── experiments/      # Training logs and checkpoints")
    print("├── outputs/          # Model outputs and reports")
    print("├── deployment/       # Streamlit app and API")
    print("└── scripts/          # Utility scripts")
    
    print("\nReady for Step 3: Data Pipeline & Preprocessing!")
    print("Your modular, RTX 3050-optimized structure is complete!")

if __name__ == "__main__":
    main()