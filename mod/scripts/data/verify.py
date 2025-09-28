import numpy as np
import json

# Quick verification
train_segments = np.load('data/processed/train/segments.npy')
train_labels = np.load('data/processed/train/labels.npy')

print(f"Training segments shape: {train_segments.shape}")
print(f"Segment length: {train_segments.shape[1]} samples (5 seconds at 360 Hz)")
print(f"Data range: [{train_segments.min():.3f}, {train_segments.max():.3f}]")

with open('data/processed/label_encoding.json') as f:
    encoding = json.load(f)
    print(f"Classes: {encoding['classes']}")