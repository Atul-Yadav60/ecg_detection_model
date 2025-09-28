#!/usr/bin/env python3
"""
Fixed PTB-XL processor - processes ALL records, not just 1000
"""

import os
import wfdb
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import ast

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def find_ptb_xl_files(data_dir):
    """Find correct PTB-XL files after extraction"""
    ptb_path = Path(data_dir) / 'ptb_xl'
    
    # Look for the actual extracted folder structure
    possible_paths = [
        ptb_path,
        ptb_path / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3',
        list(ptb_path.glob("*ptb-xl*"))[0] if list(ptb_path.glob("*ptb-xl*")) else None
    ]
    
    for path in possible_paths:
        if path and path.exists():
            csv_path = path / 'ptbxl_database.csv'
            if csv_path.exists():
                return path, csv_path
    
    raise FileNotFoundError("Could not locate PTB-XL database CSV")

def map_ptb_scp_codes(scp_codes_dict):
    """Enhanced SCP code mapping with more comprehensive coverage"""
    mapping = {
        # Normal rhythms - HIGH PRIORITY
        'NORM': 'N',    # Normal ECG
        'NSR': 'N',     # Normal sinus rhythm
        'SR': 'N',      # Sinus rhythm
        
        # Atrial Fibrillation/Flutter (TARGET F CLASS!)
        'AFIB': 'F',    # Atrial fibrillation - CRITICAL
        'AFL': 'F',     # Atrial flutter
        'SARRH': 'F',   # Sinus arrhythmia
        'AFLT': 'F',    # Atrial flutter variant
        
        # Supraventricular (S class)
        'STACH': 'S',   # Sinus tachycardia
        'SBRAD': 'S',   # Sinus bradycardia
        'SVTAC': 'S',   # Supraventricular tachycardia
        'PAC': 'S',     # Premature atrial contraction
        'SVARR': 'S',   # Supraventricular arrhythmia
        'AT': 'S',      # Atrial tachycardia
        'AVRT': 'S',    # AV reentrant tachycardia
        
        # Ventricular (V class)  
        'VT': 'V',      # Ventricular tachycardia
        'VFL': 'V',     # Ventricular flutter
        'PVC': 'V',     # Premature ventricular contraction
        'BIGEMINY': 'V',
        'TRIGEMINY': 'V',
        'VEB': 'V',     # Ventricular ectopic beats
        'VESC': 'V',    # Ventricular escape
        
        # Other/Conduction issues (Q class)
        'PACE': 'Q',    # Paced rhythm
        'LBBB': 'Q',    # Left bundle branch block
        'RBBB': 'Q',    # Right bundle branch block
        'LVH': 'Q',     # Left ventricular hypertrophy
        'RVH': 'Q',     # Right ventricular hypertrophy
        'IAVB': 'Q',    # AV blocks
        'IIAVB': 'Q',
        'IIAVBII': 'Q',
        'LVOLT': 'Q',   # Low voltage
        'HVOLT': 'Q',   # High voltage
        'LAO': 'Q',     # Left axis deviation
        'RAO': 'Q',     # Right axis deviation
        'WPW': 'Q',     # Wolff-Parkinson-White
    }
    
    if not scp_codes_dict or not isinstance(scp_codes_dict, dict):
        return None
    
    # Find the highest confidence mapped code
    best_mapping = None
    highest_confidence = -1
    
    for scp_code, confidence in scp_codes_dict.items():
        if scp_code in mapping:
            if confidence >= 0 and confidence > highest_confidence:
                best_mapping = mapping[scp_code]
                highest_confidence = confidence
    
    # Enhanced fallback with pattern matching
    if not best_mapping:
        for scp_code, confidence in scp_codes_dict.items():
            if confidence >= 30:  # Lower threshold for fallback
                # Pattern matching for unmapped codes
                scp_upper = scp_code.upper()
                if 'NORM' in scp_upper or 'NSR' in scp_upper:
                    return 'N'
                elif 'AFIB' in scp_upper or 'AFL' in scp_upper or 'AF' in scp_upper:
                    return 'F'  # CRITICAL for F class!
                elif 'BRAD' in scp_upper or 'TACH' in scp_upper:
                    return 'S'
                elif 'VT' in scp_upper or 'VEB' in scp_upper or 'PVC' in scp_upper:
                    return 'V'
                elif 'BBB' in scp_upper or 'BLOCK' in scp_upper or 'PACE' in scp_upper:
                    return 'Q'
    
    return best_mapping

def process_full_ptb_xl(data_dir="./ecg_datasets", max_records=None):
    """Process ALL PTB-XL records, not just 1000"""
    logger = setup_logging()
    
    try:
        ptb_base_path, csv_path = find_ptb_xl_files(data_dir)
        
        # Load metadata
        logger.info(f"Loading metadata from: {csv_path}")
        metadata = pd.read_csv(csv_path, index_col='ecg_id')
        
        total_records = len(metadata)
        logger.info(f"🎯 PROCESSING ALL {total_records:,} PTB-XL RECORDS")
        
        # Apply max_records limit if specified
        if max_records and max_records < total_records:
            metadata = metadata.head(max_records)
            logger.info(f"Limited to {max_records} records for testing")
        
        single_lead_data = []
        labels = []
        processed_count = 0
        file_not_found = 0
        mapping_failures = 0
        error_count = 0
        
        # Progress tracking
        batch_size = 1000
        for batch_start in range(0, len(metadata), batch_size):
            batch_end = min(batch_start + batch_size, len(metadata))
            batch_metadata = metadata.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: records {batch_start}-{batch_end}")
            
            for idx, row in tqdm(batch_metadata.iterrows(), 
                               total=len(batch_metadata), 
                               desc=f"Batch {batch_start//batch_size + 1}"):
                try:
                    # Handle filename
                    filename_lr = str(row['filename_lr'])
                    
                    # Try different path structures
                    possible_paths = [
                        ptb_base_path / filename_lr,
                        ptb_base_path / 'records500' / filename_lr,
                        ptb_base_path / 'records100' / filename_lr,
                    ]
                    
                    signal_path = None
                    for path in possible_paths:
                        if path.with_suffix('.hea').exists():
                            signal_path = path
                            break
                    
                    if not signal_path:
                        file_not_found += 1
                        continue
                    
                    # Load signal
                    signal, fields = wfdb.rdsamp(str(signal_path))
                    
                    # Extract Lead II (or first available lead)
                    if len(signal.shape) > 1 and signal.shape[1] > 1:
                        lead_ii = signal[:, 1]  # Lead II
                    else:
                        lead_ii = signal[:, 0]  # First lead
                    
                    # Process SCP codes
                    scp_codes_str = row['scp_codes']
                    
                    try:
                        scp_codes = ast.literal_eval(scp_codes_str)
                    except:
                        scp_codes = {}
                    
                    mapped_label = map_ptb_scp_codes(scp_codes)
                    
                    if mapped_label:
                        # Standardize length
                        target_length = 1000
                        if len(lead_ii) > target_length:
                            lead_ii = lead_ii[:target_length]
                        elif len(lead_ii) < target_length:
                            pad_length = target_length - len(lead_ii)
                            lead_ii = np.pad(lead_ii, (0, pad_length), 'constant', constant_values=0)
                        
                        single_lead_data.append(lead_ii)
                        labels.append(mapped_label)
                        processed_count += 1
                    else:
                        mapping_failures += 1
                        
                except Exception as e:
                    error_count += 1
                    continue
            
            # Progress update
            logger.info(f"Batch complete. Processed: {processed_count}, Failures: {file_not_found + mapping_failures + error_count}")
        
        logger.info(f"🏁 FINAL RESULTS:")
        logger.info(f"   Processed: {processed_count:,} signals")
        logger.info(f"   File not found: {file_not_found:,}")
        logger.info(f"   Mapping failures: {mapping_failures:,}")
        logger.info(f"   Other errors: {error_count:,}")
        
        if processed_count > 0:
            # Print class distribution
            from collections import Counter
            class_counts = Counter(labels)
            logger.info(f"\n📊 CLASS DISTRIBUTION:")
            for class_name, count in sorted(class_counts.items()):
                percentage = (count / len(labels)) * 100
                logger.info(f"   {class_name}: {count:,} ({percentage:.1f}%)")
            
            # Save data
            output_dir = Path(data_dir) / 'processed'
            output_dir.mkdir(exist_ok=True)
            
            np.save(output_dir / 'combined_ecg_signals.npy', np.array(single_lead_data))
            np.save(output_dir / 'combined_ecg_labels.npy', np.array(labels))
            np.save(output_dir / 'dataset_sources.npy', np.array(['ptb_xl'] * len(labels)))
            
            # Save metadata
            import json
            metadata_dict = {
                'total_samples': len(single_lead_data),
                'signal_shape': list(single_lead_data[0].shape) if single_lead_data else None,
                'unique_labels': list(class_counts.keys()),
                'class_counts': dict(class_counts),
                'datasets_used': ['ptb_xl']
            }
            
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            logger.info(f"💾 Saved processed data to {output_dir}")
            return np.array(single_lead_data), np.array(labels)
        else:
            logger.error("No data processed!")
            return np.array([]), np.array([])
            
    except Exception as e:
        logger.error(f"Critical error: {e}")
        return np.array([]), np.array([])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-records', type=int, default=None, 
                       help='Limit number of records (for testing)')
    
    args = parser.parse_args()
    
    print(f"🚀 Processing FULL PTB-XL Dataset")
    if args.max_records:
        print(f"Limited to {args.max_records:,} records for testing")
    else:
        print("Processing ALL ~21,800 records (this will take 10-15 minutes)")
    
    signals, labels = process_full_ptb_xl(max_records=args.max_records)
    
    if len(signals) > 0:
        print(f"✅ SUCCESS! Processed {len(signals):,} ECG signals")
        print("🎯 Your class F samples should be MUCH higher now!")
    else:
        print("❌ No data processed. Check error logs above.")