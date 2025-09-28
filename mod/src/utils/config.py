"""
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
