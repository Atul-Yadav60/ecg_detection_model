#!/usr/bin/env python3
"""
Ingest PTB-XL into the existing pipeline format, conservatively mapping labels.
- Uses records100 (preferred) or records500 if present under data/raw/ptb_xl
- Extracts one lead (II if available, else I)
- Resamples/crops to match existing segment length from data/processed/train/segments.npy
- Conservative label mapping:
  * NORM -> 'N'
  * else -> 'Q' (unclassifiable/other) to avoid corrupting beat-type classes
- Writes NPZ files under data/processed/additions/ptbxl_{split}.npz with keys: segments, labels
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

RAW_DIR = Path('data/raw/ptb_xl')
OUT_DIR = Path('data/processed/additions')
PROCESSED_TRAIN = Path('data/processed/train/segments.npy')
DB_CSV = RAW_DIR / 'ptbxl_database.csv'
SCP_CSV = RAW_DIR / 'scp_statements.csv'


def detect_signal_root() -> Path:
    if (RAW_DIR / 'records100').exists():
        return RAW_DIR / 'records100'
    if (RAW_DIR / 'records500').exists():
        return RAW_DIR / 'records500'
    raise FileNotFoundError('records100 or records500 not found under data/raw/ptb_xl')


def get_target_length() -> int:
    if not PROCESSED_TRAIN.exists():
        raise FileNotFoundError('data/processed/train/segments.npy not found to infer target length')
    segs = np.load(PROCESSED_TRAIN, mmap_mode='r')
    return int(segs.shape[1])


def load_metadata():
    if not DB_CSV.exists() or not SCP_CSV.exists():
        raise FileNotFoundError('ptbxl_database.csv or scp_statements.csv missing in data/raw/ptb_xl')
    df = pd.read_csv(DB_CSV)
    scp = pd.read_csv(SCP_CSV, index_col=0)
    # Map scp_codes JSON-like strings to dict
    df['scp_codes'] = df['scp_codes'].apply(lambda s: eval(s) if isinstance(s, str) else {})
    return df, scp


def map_label(scp_codes: dict) -> str:
    # PTB-XL diagnosis groups from paper: NORM, MI, STTC, CD, HYP
    if 'NORM' in scp_codes:
        return 'N'
    # Everything else marked as 'Q' to avoid polluting S/V/F beat classes
    return 'Q'


def select_lead(sig: np.ndarray, sig_names: list) -> np.ndarray:
    # Prefer lead II, else I, else first channel
    try:
        if 'II' in sig_names:
            return sig[:, sig_names.index('II')]
        if 'I' in sig_names:
            return sig[:, sig_names.index('I')]
    except Exception:
        pass
    return sig[:, 0]


def resample_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.size == target_len:
        return x.astype(np.float32)
    # Linear resample by interpolation
    src_idx = np.linspace(0.0, 1.0, num=x.size, endpoint=True)
    tgt_idx = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    y = np.interp(tgt_idx, src_idx, x)
    return y.astype(np.float32)


def main():
    print('🔎 Ingesting PTB-XL...')
    sig_root = detect_signal_root()
    target_len = get_target_length()
    df, scp = load_metadata()

    # Choose filename column according to folder present
    fname_col = 'filename_lr' if sig_root.name == 'records100' else 'filename_hr'

    segments = []
    labels = []

    total = len(df)
    for idx, row in df.iterrows():
        rel = row[fname_col]
        path = RAW_DIR / rel
        if not path.exists():
            continue
        try:
            rec = wfdb.rdrecord(str(path.with_suffix('')))
            sig = np.asarray(rec.p_signal)
            lead = select_lead(sig, list(rec.sig_name))
            lead = resample_to_length(lead, target_len)
            label = map_label(row['scp_codes'])
            segments.append(lead)
            labels.append(label)
        except Exception:
            continue

        if (idx + 1) % 1000 == 0:
            print(f'... processed {idx+1}/{total}')

    if not segments:
        raise RuntimeError('No PTB-XL samples ingested. Ensure files are present under data/raw/ptb_xl')

    segments = np.stack(segments, axis=0)
    labels = np.array(labels, dtype=object)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'ptbxl_train_like.npz'
    np.savez_compressed(out_path, segments=segments, labels=labels)

    # Report distribution
    unique, counts = np.unique(labels, return_counts=True)
    print('✅ PTB-XL ingestion complete:', out_path)
    for u, c in zip(unique, counts):
        print(f'   {u}: {c} samples')


if __name__ == '__main__':
    main()
