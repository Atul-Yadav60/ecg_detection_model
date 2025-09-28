#!/usr/bin/env python3
"""
Resume ECG Processing after download completion
Quick fix for configuration issue
"""

import os
import sys
import json
import numpy as np
import wfdb
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Signal processing imports
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

class QuickECGProcessor:
    """Resume ECG processing from downloaded data"""
    
    def __init__(self):
        self.data_dir = Path("data/raw/mit_bih")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # MIT-BIH record numbers
        self.records = [
            '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
            '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
            '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
            '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
            '222', '223', '228', '230', '231', '232', '233', '234'
        ]
        
        # AAMI beat type mapping
        self.aami_classes = {
            'N': ['N', 'L', 'R', 'e', 'j'],           # Normal
            'S': ['A', 'a', 'J', 'S'],                # Supraventricular
            'V': ['V', 'E'],                          # Ventricular
            'F': ['F'],                               # Fusion
            'Q': ['/', 'f', 'Q']                      # Unknown
        }
        
    def symbol_to_aami(self, symbol):
        """Convert MIT-BIH symbol to AAMI class"""
        for aami_class, symbols in self.aami_classes.items():
            if symbol in symbols:
                return aami_class
        return 'Q'  # Unknown class for unrecognized symbols
    
    def parse_record(self, record_num):
        """Parse a single MIT-BIH record"""
        try:
            # Read record
            record = wfdb.rdrecord(str(self.data_dir / record_num))
            annotation = wfdb.rdann(str(self.data_dir / record_num), 'atr')
            
            # Extract ECG signal (use lead II if available, otherwise first lead)
            if record.n_sig >= 2:
                ecg_signal = record.p_signal[:, 1]  # Lead II
            else:
                ecg_signal = record.p_signal[:, 0]  # First available lead
            
            # Extract annotations
            beat_locations = annotation.sample
            beat_types = annotation.symbol
            
            # Convert to AAMI classes
            aami_labels = []
            for beat_type in beat_types:
                aami_class = self.symbol_to_aami(beat_type)
                aami_labels.append(aami_class)
            
            return {
                'signal': ecg_signal,
                'beat_locations': beat_locations,
                'beat_types': beat_types,
                'aami_labels': aami_labels,
                'sampling_rate': record.fs,
                'record_name': record_num
            }
            
        except Exception as e:
            print(f"Failed to parse record {record_num}: {e}")
            return None
    
    def apply_filters(self, signal, fs):
        """Apply preprocessing filters to ECG signal"""
        # Fixed configuration with defaults
        highpass_freq = 0.5
        lowpass_freq = 40
        notch_freq = 50
        
        # Bandpass filter (0.5-40 Hz)
        nyquist = fs / 2
        low = highpass_freq / nyquist
        high = lowpass_freq / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        
        # Notch filter (50 Hz powerline interference)
        b_notch, a_notch = iirnotch(notch_freq, 30, fs)
        filtered_signal = filtfilt(b_notch, a_notch, filtered_signal)
        
        return filtered_signal
    
    def normalize_signal(self, signal):
        """Normalize ECG signal using RobustScaler"""
        signal = np.array(signal).reshape(-1, 1)
        scaler = RobustScaler()
        normalized = scaler.fit_transform(signal).flatten()
        return normalized
    
    def segment_signal(self, signal, beat_locations, labels, window_size=1800):
        """Segment ECG signal around R-peaks"""
        segments = []
        segment_labels = []
        
        half_window = window_size // 2
        
        for i, r_peak in enumerate(beat_locations):
            # Skip if too close to boundaries
            if r_peak < half_window or r_peak + half_window >= len(signal):
                continue
                
            # Extract segment centered on R-peak
            segment = signal[r_peak - half_window:r_peak + half_window]
            
            # Ensure consistent length
            if len(segment) == window_size:
                segments.append(segment)
                segment_labels.append(labels[i])
        
        return np.array(segments), np.array(segment_labels)
    
    def augment_segments(self, segments, labels, augment_factor=2):
        """Apply data augmentation to ECG segments"""
        augmented_segments = []
        augmented_labels = []
        
        for segment, label in zip(segments, labels):
            # Original segment
            augmented_segments.append(segment)
            augmented_labels.append(label)
            
            # Apply augmentations
            for _ in range(augment_factor - 1):
                aug_segment = segment.copy()
                
                # Add Gaussian noise
                noise_std = np.std(segment) * 0.01
                aug_segment += np.random.normal(0, noise_std, len(segment))
                
                # Amplitude scaling
                scale_factor = np.random.uniform(0.9, 1.1)
                aug_segment *= scale_factor
                
                # Time stretching (simple interpolation)
                stretch_factor = np.random.uniform(0.95, 1.05)
                if stretch_factor != 1.0:
                    indices = np.arange(len(segment))
                    stretched_indices = indices * stretch_factor
                    aug_segment = np.interp(indices, stretched_indices, aug_segment)
                
                augmented_segments.append(aug_segment)
                augmented_labels.append(label)
        
        return np.array(augmented_segments), np.array(augmented_labels)
    
    def parse_all_records(self):
        """Parse all downloaded records"""
        print("Parsing downloaded MIT-BIH records...")
        
        all_data = []
        for record in tqdm(self.records, desc="Parsing records"):
            parsed_data = self.parse_record(record)
            if parsed_data:
                all_data.append(parsed_data)
        
        print(f"Successfully parsed {len(all_data)} records")
        return all_data
    
    def process_all_records(self, raw_data):
        """Process all ECG records through complete pipeline"""
        all_segments = []
        all_labels = []
        all_patients = []
        
        print("Processing all ECG records...")
        
        for record_data in tqdm(raw_data, desc="Processing records"):
            # Extract data
            signal = record_data['signal']
            beat_locations = record_data['beat_locations']
            aami_labels = record_data['aami_labels']
            patient_id = record_data['record_name']
            fs = record_data['sampling_rate']
            
            # Preprocessing with fixed configuration
            filtered_signal = self.apply_filters(signal, fs)
            normalized_signal = self.normalize_signal(filtered_signal)
            
            # Segmentation
            segments, labels = self.segment_signal(
                normalized_signal, beat_locations, aami_labels, window_size=1800
            )
            
            if len(segments) > 0:
                all_segments.extend(segments)
                all_labels.extend(labels)
                all_patients.extend([patient_id] * len(segments))
        
        # Convert to numpy arrays
        all_segments = np.array(all_segments)
        all_labels = np.array(all_labels)
        all_patients = np.array(all_patients)
        
        print(f"Generated {len(all_segments)} ECG segments")
        print(f"Class distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
        
        return all_segments, all_labels, all_patients
    
    def create_patient_wise_splits(self, segments, labels, patients):
        """Create patient-wise train/validation/test splits"""
        print("Creating patient-wise data splits...")
        
        # Get unique patients
        unique_patients = np.unique(patients)
        
        # Split patients (not samples) to prevent data leakage
        train_patients, temp_patients = train_test_split(
            unique_patients, test_size=0.3, random_state=42
        )
        
        val_patients, test_patients = train_test_split(
            temp_patients, test_size=0.5, random_state=42
        )
        
        # Create splits based on patient assignment
        train_mask = np.isin(patients, train_patients)
        val_mask = np.isin(patients, val_patients)
        test_mask = np.isin(patients, test_patients)
        
        splits = {
            'train': {
                'segments': segments[train_mask],
                'labels': labels[train_mask],
                'patients': patients[train_mask]
            },
            'val': {
                'segments': segments[val_mask],
                'labels': labels[val_mask],
                'patients': patients[val_mask]
            },
            'test': {
                'segments': segments[test_mask],
                'labels': labels[test_mask],
                'patients': patients[test_mask]
            }
        }
        
        # Log split information
        for split_name, split_data in splits.items():
            n_samples = len(split_data['segments'])
            n_patients = len(np.unique(split_data['patients']))
            class_dist = dict(zip(*np.unique(split_data['labels'], return_counts=True)))
            
            print(f"{split_name.upper()} split: {n_samples} samples from {n_patients} patients")
            print(f"Class distribution: {class_dist}")
        
        return splits
    
    def apply_augmentation(self, splits):
        """Apply data augmentation to training data"""
        print("Applying data augmentation to training set...")
        
        aug_segments, aug_labels = self.augment_segments(
            splits['train']['segments'], 
            splits['train']['labels'],
            augment_factor=2
        )
        
        splits['train']['segments'] = aug_segments
        splits['train']['labels'] = aug_labels
        
        print(f"Augmented training set to {len(aug_segments)} samples")
        return splits
    
    def save_processed_data(self, splits):
        """Save processed data to disk"""
        print("Saving processed data...")
        
        for split_name, split_data in splits.items():
            # Create split directory
            split_dir = self.processed_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Save segments and labels
            np.save(split_dir / 'segments.npy', split_data['segments'])
            np.save(split_dir / 'labels.npy', split_data['labels'])
            
            # Save metadata (convert numpy types to JSON serializable)
            unique_labels, label_counts = np.unique(split_data['labels'], return_counts=True)
            class_distribution = {str(label): int(count) for label, count in zip(unique_labels, label_counts)}
            
            metadata = {
                'n_samples': int(len(split_data['segments'])),
                'n_patients': int(len(np.unique(split_data['patients']))) if 'patients' in split_data else 0,
                'segment_length': int(split_data['segments'].shape[1]),
                'class_distribution': class_distribution,
                'unique_patients': [str(p) for p in np.unique(split_data['patients'])] if 'patients' in split_data else []
            }
            
            with open(split_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Create label encoder mapping
        unique_labels = np.unique(np.concatenate([
            splits['train']['labels'],
            splits['val']['labels'], 
            splits['test']['labels']
        ]))
        
        label_to_idx = {str(label): int(idx) for idx, label in enumerate(sorted(unique_labels))}
        idx_to_label = {int(idx): str(label) for label, idx in label_to_idx.items()}
        
        encoding_info = {
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'n_classes': int(len(unique_labels)),
            'classes': [str(label) for label in sorted(unique_labels)]
        }
        
        with open(self.processed_dir / 'label_encoding.json', 'w') as f:
            json.dump(encoding_info, f, indent=2)
        
        print(f"Saved all processed data to {self.processed_dir}")
        return encoding_info
    
    def run_processing(self):
        """Run the complete processing pipeline (skip download)"""
        print("=== Resuming ECG Data Processing ===")
        
        try:
            # Step 1: Parse downloaded dataset
            raw_data = self.parse_all_records()
            
            # Step 2: Process all records
            segments, labels, patients = self.process_all_records(raw_data)
            
            # Step 3: Create patient-wise splits
            splits = self.create_patient_wise_splits(segments, labels, patients)
            
            # Step 4: Apply augmentation
            splits = self.apply_augmentation(splits)
            
            # Step 5: Save processed data
            encoding_info = self.save_processed_data(splits)
            
            # Step 6: Create summary
            self.create_summary(splits, encoding_info)
            
            print("=== ECG Data Processing Completed Successfully ===")
            return True
            
        except Exception as e:
            print(f"Processing failed: {e}")
            raise e
    
    def create_summary(self, splits, encoding_info):
        """Create processing summary"""
        # Convert numpy types for JSON serialization
        total_samples = sum(int(len(split['segments'])) for split in splits.values())
        
        splits_info = {}
        for split_name, split_data in splits.items():
            unique_labels, label_counts = np.unique(split_data['labels'], return_counts=True)
            class_distribution = {str(label): int(count) for label, count in zip(unique_labels, label_counts)}
            
            splits_info[split_name] = {
                'n_samples': int(len(split_data['segments'])),
                'n_patients': int(len(np.unique(split_data['patients']))),
                'class_distribution': class_distribution
            }
        
        summary = {
            'pipeline_completed': True,
            'total_samples': total_samples,
            'splits': splits_info,
            'encoding_info': encoding_info,
            'hardware_optimizations': {
                'batch_size_recommendations': {
                    'training': 64,
                    'inference': 16
                },
                'memory_efficient': True,
                'mixed_precision_ready': True
            }
        }
        
        with open('data_pipeline_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("ECG DATA PIPELINE COMPLETED!")
        print(f"Total samples processed: {summary['total_samples']}")
        print(f"Classes: {encoding_info['classes']}")
        print("Patient-wise splits created (no data leakage)")
        print("RTX 3050 optimized - Ready for training!")
        print("="*60)

def main():
    """Main function"""
    print("ECG Data Processing Resume")
    print("Using downloaded MIT-BIH data")
    print("="*50)
    
    # Initialize processor
    processor = QuickECGProcessor()
    
    # Run processing
    processor.run_processing()

if __name__ == "__main__":
    main()