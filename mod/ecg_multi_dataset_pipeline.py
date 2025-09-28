#!/usr/bin/env python3
"""
Multi-Dataset ECG Download and Preprocessing Pipeline - COMPLETE FIXED VERSION
Handles different dataset structures and file paths correctly
"""

import os
import requests
import wfdb
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import zipfile
import ast
from sklearn.preprocessing import LabelEncoder

class ECGMultiDatasetLoader:
    def __init__(self, data_dir="./ecg_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def inspect_dataset_structure(self, dataset_name):
        """Inspect downloaded dataset structure to find correct files"""
        dataset_path = self.data_dir / dataset_name
        
        self.logger.info(f"Inspecting {dataset_name} structure:")
        
        # Find all CSV files
        csv_files = list(dataset_path.rglob("*.csv"))
        self.logger.info(f"CSV files found: {csv_files}")
        
        # Find all directories
        dirs = [d for d in dataset_path.rglob("*") if d.is_dir()]
        self.logger.info(f"Directories: {dirs[:5]}")  # Show first 5
        
        return csv_files, dirs
    
    def find_ptb_xl_files(self):
        """Find correct PTB-XL files after extraction"""
        ptb_path = self.data_dir / 'ptb_xl'
        
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
                    self.logger.info(f"Found PTB-XL database at: {csv_path}")
                    return path, csv_path
        
        # If not found, inspect structure
        csv_files, dirs = self.inspect_dataset_structure('ptb_xl')
        
        # Try to find database CSV by name
        for csv_file in csv_files:
            if 'database' in csv_file.name.lower() or 'ptbxl' in csv_file.name.lower():
                self.logger.info(f"Found database CSV: {csv_file}")
                return csv_file.parent, csv_file
                
        raise FileNotFoundError(f"Could not locate PTB-XL database CSV. Files found: {csv_files}")
    
    def process_ptb_xl(self):
        """Fixed PTB-XL processing with debugging and better path handling"""
        try:
            ptb_base_path, csv_path = self.find_ptb_xl_files()
            
            # Load metadata
            self.logger.info(f"Loading metadata from: {csv_path}")
            metadata = pd.read_csv(csv_path, index_col='ecg_id')
            
            self.logger.info(f"Loaded {len(metadata)} records from PTB-XL")
            self.logger.info(f"Available columns: {list(metadata.columns)}")
            
            # DEBUG: Show sample SCP codes to understand format
            sample_scp = metadata['scp_codes'].iloc[:5]
            self.logger.info(f"Sample SCP codes: {sample_scp.tolist()}")
            
            single_lead_data = []
            labels = []
            processed_count = 0
            error_count = 0
            mapping_failures = 0
            file_not_found = 0
            
            # Process subset first to test
            subset_size = min(1000, len(metadata))
            metadata_subset = metadata.head(subset_size)
            
            self.logger.info(f"Processing first {subset_size} records as test")
            
            for idx, row in tqdm(metadata_subset.iterrows(), total=len(metadata_subset)):
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
                        # DEBUG first few file not found
                        if file_not_found <= 3:
                            self.logger.debug(f"File not found for: {filename_lr}")
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
                    
                    mapped_label = self.map_ptb_scp_codes(scp_codes)
                    
                    # DEBUG first few mappings
                    if processed_count < 5:
                       self.logger.info(f"DEBUG: Signal shape: {signal.shape}, Lead II length: {len(lead_ii)}, Sampling rate from header: {fields.get('fs', 'unknown')}")
                    
                    if mapped_label:
                        # Standardize length (2.5 seconds at 500Hz = 1250 samples)
                        target_length = 1000
                        min_length = 500
                        if len(lead_ii) >= min_length:
                          if len(lead_ii) > target_length:
                             lead_ii = lead_ii[:target_length]
                        elif len(lead_ii) < target_length:
                            # Pad with zeros
                         pad_length = target_length - len(lead_ii)
                         lead_ii = np.pad(lead_ii, (0, pad_length), 'constant', constant_values=0)
                        single_lead_data.append(lead_ii)
                        labels.append(mapped_label)
                        processed_count += 1
                    else:
                        mapping_failures += 1
                        
                except Exception as e:
                    error_count += 1
                    # DEBUG first few errors
                    if error_count <= 3:
                        self.logger.error(f"Error processing record {idx}: {e}")
                    continue
            
            self.logger.info(f"PTB-XL Complete: {processed_count} signals")
            self.logger.info(f"File not found: {file_not_found}")
            self.logger.info(f"Mapping failures: {mapping_failures}")
            self.logger.info(f"Other errors: {error_count}")
            
            if processed_count > 0:
                return np.array(single_lead_data), np.array(labels)
            else:
                return np.array([]), np.array([])
                
        except Exception as e:
            self.logger.error(f"Critical error in PTB-XL processing: {e}")
            return np.array([]), np.array([])
    
    def map_ptb_scp_codes(self, scp_codes_dict):
        """Map PTB-XL SCP codes to your classes (F, N, Q, S, V)"""
        # Based on debug output - PTB-XL uses text SCP codes with confidence scores
        mapping = {
            # Normal rhythms - HIGH PRIORITY
            'NORM': 'N',    # Normal ECG (seen in debug!)
            'NSR': 'N',     # Normal sinus rhythm
            'SR': 'N',      # Sinus rhythm (seen in debug!)
            
            # Atrial Fibrillation/Flutter (TARGET F CLASS!)
            'AFIB': 'F',    # Atrial fibrillation - CRITICAL
            'AFL': 'F',     # Atrial flutter
            'SARRH': 'F',   # Sinus arrhythmia
            
            # Supraventricular (S class) - Including bradycardia/tachycardia
            'STACH': 'S',   # Sinus tachycardia
            'SBRAD': 'S',   # Sinus bradycardia (seen in debug!)
            'SVTAC': 'S',   # Supraventricular tachycardia
            'PAC': 'S',     # Premature atrial contraction
            'SVARR': 'S',   # Supraventricular arrhythmia
            
            # Ventricular (V class)  
            'VT': 'V',      # Ventricular tachycardia
            'VFL': 'V',     # Ventricular flutter
            'PVC': 'V',     # Premature ventricular contraction
            'BIGEMINY': 'V',
            'TRIGEMINY': 'V',
            'VEB': 'V',     # Ventricular ectopic beats
            
            # Other/Conduction issues (Q class)
            'PACE': 'Q',    # Paced rhythm
            'LBBB': 'Q',    # Left bundle branch block
            'RBBB': 'Q',    # Right bundle branch block
            'LVH': 'Q',     # Left ventricular hypertrophy
            'RVH': 'Q',     # Right ventricular hypertrophy
            'IAVB': 'Q',    # AV blocks
            'IIAVB': 'Q',
            'IIAVB': 'Q',
            'LVOLT': 'Q',   # Low voltage (seen in debug!)
        }
        
        # Check if scp_codes is valid
        if not scp_codes_dict or not isinstance(scp_codes_dict, dict):
            return None
        
        # Find the highest confidence mapped code
        best_mapping = None
        highest_confidence = -1
        
        for scp_code, confidence in scp_codes_dict.items():
            if scp_code in mapping:
                # Use any confidence > 0, but prefer higher confidence
                if confidence >= 0 and confidence > highest_confidence:
                    best_mapping = mapping[scp_code]
                    highest_confidence = confidence
        
        # If we found a mapping, return it
        if best_mapping:
            return best_mapping
        
        # Fallback: If no direct mapping found, use any code with reasonable confidence
        # This ensures we don't lose data due to strict mapping
        for scp_code, confidence in scp_codes_dict.items():
            if confidence >= 50:  # High confidence codes
                # Try pattern matching
                if 'NORM' in scp_code or 'SR' in scp_code:
                    return 'N'
                elif 'BRAD' in scp_code or 'TACH' in scp_code:
                    return 'S'  # Rhythm variations
                elif 'VT' in scp_code or 'VEB' in scp_code:
                    return 'V'
                elif 'AF' in scp_code:  # Any AF variant
                    return 'F'
        
        return None
    
    def download_dataset(self, dataset_name, force_download=False):
        """Download and extract dataset with better error handling"""
        dataset_info = self.datasets.get(dataset_name)
        if not dataset_info:
            self.logger.error(f"Unknown dataset: {dataset_name}")
            return None
            
        dataset_path = self.data_dir / dataset_name
        
        if dataset_path.exists() and not force_download:
            self.logger.info(f"{dataset_name} already exists. Skipping download.")
            return dataset_path
            
        self.logger.info(f"Downloading {dataset_name} ({dataset_info['size']})")
        
        try:
            # Download with timeout and error handling
            response = requests.get(dataset_info['url'], stream=True, timeout=300)
            response.raise_for_status()
            
            zip_path = self.data_dir / f"{dataset_name}.zip"
            
            # Download with progress bar
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                desc=f"Downloading {dataset_name}",
                total=total_size,
                unit='B',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
                    
            # Extract
            self.logger.info(f"Extracting {dataset_name}")
            dataset_path.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
                
            # Cleanup
            zip_path.unlink()
            
            self.logger.info(f"Successfully downloaded and extracted {dataset_name}")
            return dataset_path
            
        except Exception as e:
            self.logger.error(f"Error downloading {dataset_name}: {e}")
            return None
    
    def process_mitdb_simple(self):
        """Simplified MIT-BIH processing"""
        mitdb_path = self.data_dir / 'mitdb'
        
        # Find actual MIT-BIH files
        dat_files = list(mitdb_path.rglob("*.dat"))
        
        if not dat_files:
            self.logger.error("No MIT-BIH .dat files found")
            return np.array([]), np.array([])
        
        self.logger.info(f"Found {len(dat_files)} MIT-BIH records")
        
        all_signals = []
        all_labels = []
        
        # Beat type mapping to your classes
        beat_mapping = {
            'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',  # Normal types
            'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',            # Supraventricular
            'V': 'V', 'E': 'V',                                 # Ventricular
            'F': 'F',                                           # Fusion
            '/': 'Q', 'f': 'Q', 'Q': 'Q',                     # Other/Unclassified
        }
        
        for dat_file in tqdm(dat_files[:5], desc="Processing MIT-BIH"):  # Start with first 5
            try:
                record_name = dat_file.stem
                record_path = dat_file.parent / record_name
                
                # Load signal and annotations
                signal, fields = wfdb.rdsamp(str(record_path))
                
                # Use first channel if multiple
                if len(signal.shape) > 1:
                    ecg_signal = signal[:, 0]
                else:
                    ecg_signal = signal
                
                # Load annotations
                try:
                    annotations = wfdb.rdann(str(record_path), 'atr')
                    
                    # Segment around each beat
                    for i, (sample, symbol) in enumerate(zip(annotations.sample, annotations.symbol)):
                        if symbol in beat_mapping:
                            # Extract 2-second window around beat
                            window_size = 720  # 2 seconds at 360Hz
                            start = max(0, sample - window_size//2)
                            end = min(len(ecg_signal), sample + window_size//2)
                            
                            if end - start == window_size:
                                segment = ecg_signal[start:end]
                                all_signals.append(segment)
                                all_labels.append(beat_mapping[symbol])
                                
                except Exception as e:
                    self.logger.warning(f"Could not load annotations for {record_name}: {e}")
                    continue
                    
            except Exception as e:
                self.logger.warning(f"Error processing {dat_file}: {e}")
                continue
        
        self.logger.info(f"MIT-BIH processing complete: {len(all_signals)} segments")
        return np.array(all_signals), np.array(all_labels)
    
    def process_datasets_safely(self, dataset_list):
        """Process datasets with error recovery"""
        all_signals = []
        all_labels = []
        dataset_sources = []
        
        for dataset_name in dataset_list:
            self.logger.info(f"Processing {dataset_name}")
            
            try:
                if dataset_name == 'ptb_xl':
                    signals, labels = self.process_ptb_xl()
                elif dataset_name == 'mitdb':
                    signals, labels = self.process_mitdb_simple()
                else:
                    self.logger.info(f"Skipping {dataset_name} for now")
                    continue
                
                if len(signals) > 0:
                    all_signals.extend(signals)
                    all_labels.extend(labels)
                    dataset_sources.extend([dataset_name] * len(signals))
                    
                    self.logger.info(f"Successfully added {len(signals)} samples from {dataset_name}")
                else:
                    self.logger.warning(f"No valid samples from {dataset_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {dataset_name}: {e}")
                self.logger.info(f"Continuing with other datasets...")
                continue
        
        return all_signals, all_labels, dataset_sources
    
    def print_class_distribution(self, labels):
        """Print detailed class distribution"""
        if len(labels) == 0:
            print("No labels to analyze")
            return
            
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        print("\n📊 COMBINED DATASET CLASS DISTRIBUTION:")
        print("="*50)
        for class_name, count in zip(unique, counts):
            percentage = (count / total) * 100
            print(f"   {class_name}: {count:,} samples ({percentage:.1f}%)")
        print(f"\nTotal samples: {total:,}")
        
        # Calculate F class improvement
        if 'F' in unique:
            f_samples = counts[list(unique).index('F')]
            original_f = 842  # Your original F samples
            improvement = f_samples / original_f
            print(f"\n🎯 Class F Improvement: {improvement:.1f}x more samples!")
    
    def save_processed_data(self, signals, labels, sources):
        """Save processed data with metadata"""
        if len(signals) == 0:
            self.logger.error("No data to save")
            return
        
        # Create output directory
        output_dir = self.data_dir / 'processed'
        output_dir.mkdir(exist_ok=True)
        
        # Save arrays
        np.save(output_dir / 'combined_ecg_signals.npy', signals)
        np.save(output_dir / 'combined_ecg_labels.npy', labels)
        np.save(output_dir / 'dataset_sources.npy', sources)
        
        # Save metadata
        metadata = {
            'total_samples': len(signals),
            'signal_shape': signals[0].shape if len(signals) > 0 else None,
            'unique_labels': list(np.unique(labels)),
            'datasets_used': list(np.unique(sources))
        }
        
        import json
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved processed data to {output_dir}")

def main():
    """Main function with command line argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ECG Multi-Dataset Pipeline')
    parser.add_argument('--datasets', nargs='+', default=['mitdb'], 
                       help='Datasets to process')
    parser.add_argument('--quick-start', action='store_true',
                       help='Process smaller subset for testing')
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = ECGMultiDatasetLoader()
    
    print("🚀 Starting Multi-Dataset ECG Pipeline")
    print(f"Selected datasets: {args.datasets}")
    
    # Process datasets
    try:
        all_signals, all_labels, sources = loader.process_datasets_safely(args.datasets)
        
        if len(all_signals) > 0:
            # Convert to numpy arrays
            combined_signals = np.array(all_signals)
            combined_labels = np.array(all_labels)
            
            # Print results
            loader.print_class_distribution(combined_labels)
            
            # Save processed data
            loader.save_processed_data(combined_signals, combined_labels, sources)
            
            print("✅ Multi-dataset pipeline complete!")
            print(f"📁 Data saved to: {loader.data_dir}/processed/")
            
        else:
            print("❌ No valid data processed. Check logs for errors.")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

# Add datasets info after class definition
ECGMultiDatasetLoader.datasets = {
    'ptb_xl': {
        'url': 'https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip',
        'size': '21,801 records',
        'leads': 12,
        'priority': 'HIGH'
    },
    'mitdb': {
        'url': 'https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip',
        'size': '48 records',
        'leads': 2,
        'priority': 'HIGH'  # Start here for testing
    }
}

if __name__ == "__main__":
    main()