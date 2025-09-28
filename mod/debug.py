#!/usr/bin/env python3
"""
Debug script to understand PTB-XL structure and SCP codes
"""

import pandas as pd
import ast
from pathlib import Path
import os

def debug_ptb_xl():
    # Load the database
    csv_path = Path('ecg_datasets/ptb_xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv')
    
    print("🔍 DEBUGGING PTB-XL DATASET")
    print("=" * 50)
    
    # Load first 10 records
    df = pd.read_csv(csv_path, nrows=10)
    
    print("1. SAMPLE SCP CODES:")
    print("-" * 30)
    for i in range(5):
        scp_raw = df['scp_codes'].iloc[i]
        filename = df['filename_lr'].iloc[i]
        print(f"Record {i+1}:")
        print(f"   Filename: {filename}")
        print(f"   SCP Raw: {scp_raw}")
        try:
            scp_parsed = ast.literal_eval(scp_raw)
            print(f"   SCP Parsed: {scp_parsed}")
            print(f"   SCP Keys: {list(scp_parsed.keys())}")
        except Exception as e:
            print(f"   Parse Error: {e}")
        print()
    
    print("2. FILE PATH TESTING:")
    print("-" * 30)
    base_path = Path('ecg_datasets/ptb_xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
    
    # Check first filename
    first_filename = df['filename_lr'].iloc[0]
    print(f"Testing filename: {first_filename}")
    
    possible_paths = [
        base_path / first_filename,
        base_path / 'records500' / first_filename,
        base_path / 'records100' / first_filename,
    ]
    
    for path in possible_paths:
        hea_path = path.with_suffix('.hea')
        dat_path = path.with_suffix('.dat')
        print(f"   Path: {path}")
        print(f"   .hea exists: {hea_path.exists()}")
        print(f"   .dat exists: {dat_path.exists()}")
        if hea_path.exists():
            print(f"   ✅ FOUND VALID PATH!")
            break
    
    print("\n3. DIRECTORY STRUCTURE:")
    print("-" * 30)
    print("Records500 contents (first 5):")
    records500_path = base_path / 'records500'
    if records500_path.exists():
        subdirs = [d for d in records500_path.iterdir() if d.is_dir()][:5]
        for subdir in subdir:
            files = list(subdir.glob('*.hea'))[:3]
            print(f"   {subdir.name}/: {len(files)} .hea files")
    
    print("\n4. SCP CODE ANALYSIS:")
    print("-" * 30)
    
    # Load more records to analyze SCP codes
    df_large = pd.read_csv(csv_path, nrows=1000)
    
    all_scp_codes = set()
    for scp_str in df_large['scp_codes']:
        try:
            scp_dict = ast.literal_eval(scp_str)
            all_scp_codes.update(scp_dict.keys())
        except:
            continue
    
    print(f"Unique SCP codes found: {sorted(list(all_scp_codes))}")
    
    print("\n5. LOOKING FOR TARGET CLASSES:")
    print("-" * 30)
    target_codes = ['AFIB', 'AFL', 'NORM', 'STACH', 'SBRAD', 'VT', 'PVC']
    
    for target in target_codes:
        count = 0
        for scp_str in df_large['scp_codes']:
            try:
                scp_dict = ast.literal_eval(scp_str)
                if target in scp_dict:
                    count += 1
            except:
                continue
        print(f"   {target}: {count} records")

if __name__ == "__main__":
    debug_ptb_xl()