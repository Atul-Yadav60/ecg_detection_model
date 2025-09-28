#!/usr/bin/env python3
"""
Merge new ingested datasets into existing train/val/test arrays.
- Loads additions from data/processed/additions/*.npz (e.g., PTB-XL)
- Appends to training set only by default (configurable)
- Re-encodes labels using existing label encoder mapping
- Saves updated arrays back to data/processed/{split}/
"""

import os
from pathlib import Path
import numpy as np
import json

BASE = Path('data/processed')
ADDITIONS = Path('data/processed/additions')
ENCODER_JSON = BASE / 'label_encoding.json'


def load_split(split: str):
    segs = np.load(BASE / split / 'segments.npy')
    labels = np.load(BASE / split / 'labels.npy')
    return segs, labels


def save_split(split: str, segs: np.ndarray, labels: np.ndarray):
    out_dir = BASE / split
    np.save(out_dir / 'segments.npy', segs)
    np.save(out_dir / 'labels.npy', labels)


def load_encoder():
    with open(ENCODER_JSON, 'r') as f:
        enc = json.load(f)
    # enc: {"N": 0, "S": 1, ...}
    return enc


def encode_labels(str_labels: np.ndarray, enc: dict) -> np.ndarray:
    # Unknown labels become 'Q'
    out = []
    for s in str_labels:
        label = s if s in enc else 'Q'
        out.append(enc[label])
    return np.array(out, dtype=np.int64)


def main():
    print('🔗 Merging additions into existing datasets...')
    train_segs, train_labels = load_split('train')
    val_segs, val_labels = load_split('val')
    test_segs, test_labels = load_split('test')
    enc = load_encoder()

    additions = sorted(ADDITIONS.glob('*.npz'))
    if not additions:
        raise FileNotFoundError('No additions found in data/processed/additions/*.npz')

    added = 0
    for npz in additions:
        data = np.load(npz, allow_pickle=True)
        segs = data['segments']
        labels = data['labels']
        # Basic sanity: length match
        if segs.shape[1] != train_segs.shape[1]:
            print(f'⚠️ Skipping {npz.name}: segment length mismatch ({segs.shape[1]} vs {train_segs.shape[1]})')
            continue
        enc_labels = encode_labels(labels, enc)
        # Append to train set only
        train_segs = np.concatenate([train_segs, segs.astype(train_segs.dtype)], axis=0)
        train_labels = np.concatenate([train_labels, enc_labels.astype(train_labels.dtype)], axis=0)
        added += segs.shape[0]
        print(f'➕ Added {segs.shape[0]} samples from {npz.name} to train')

    save_split('train', train_segs, train_labels)

    print('✅ Merge complete')
    print(f'   Train size: {train_segs.shape[0]} samples')
    print(f'   Val size:   {val_segs.shape[0]} samples')
    print(f'   Test size:  {test_segs.shape[0]} samples')
    print(f'   Added:      {added} samples')


if __name__ == '__main__':
    main()
