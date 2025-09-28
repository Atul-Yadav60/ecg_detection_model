#!/usr/bin/env python3
"""
Complete ECG Dataset Analysis Tool
Shows original dataset + new PTB-XL data + combined totals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter
import os

class ECGDatasetAnalyzer:
    def __init__(self):
        self.original_data = None
        self.new_data = None
        self.combined_data = None
        
    def load_original_dataset(self, data_path="./data"):
        """Load your original ECG dataset"""
        try:
            data_dir = Path(data_path)
            
            # Try different possible file names for your original data
            possible_files = [
                'X_train.npy', 'y_train.npy',
                'ecg_signals.npy', 'ecg_labels.npy',
                'X_processed.npy', 'y_processed.npy',
                'train_signals.npy', 'train_labels.npy'
            ]
            
            # Look for original data files
            signal_file = None
            label_file = None
            
            for file in data_dir.glob("*.npy"):
                filename = file.name.lower()
                if 'signal' in filename or 'x_' in filename:
                    signal_file = file
                elif 'label' in filename or 'y_' in filename:
                    label_file = file
            
            if signal_file and label_file:
                signals = np.load(signal_file)
                labels = np.load(label_file)
                
                print(f"📂 Found original dataset:")
                print(f"   Signals: {signal_file.name} - Shape: {signals.shape}")
                print(f"   Labels: {label_file.name} - Shape: {labels.shape}")
                
                self.original_data = {
                    'signals': signals,
                    'labels': labels,
                    'source': 'Original Dataset',
                    'files': [signal_file.name, label_file.name]
                }
                return True
            else:
                print(f"❌ Original dataset not found in {data_path}")
                print(f"Available .npy files: {[f.name for f in data_dir.glob('*.npy')]}")
                return False
                
        except Exception as e:
            print(f"Error loading original dataset: {e}")
            return False
    
    def load_new_ptb_dataset(self, data_path="./ecg_datasets/processed"):
        """Load your new PTB-XL processed dataset"""
        try:
            data_dir = Path(data_path)
            
            if not data_dir.exists():
                print(f"❌ New PTB-XL data not found at {data_path}")
                return False
            
            # Load new processed data
            signals = np.load(data_dir / 'combined_ecg_signals.npy')
            labels = np.load(data_dir / 'combined_ecg_labels.npy')
            
            # Load metadata if available
            metadata_file = data_dir / 'metadata.json'
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            print(f"📂 Found new PTB-XL dataset:")
            print(f"   Signals shape: {signals.shape}")
            print(f"   Labels shape: {labels.shape}")
            
            self.new_data = {
                'signals': signals,
                'labels': labels,
                'source': 'PTB-XL Dataset',
                'metadata': metadata
            }
            return True
            
        except Exception as e:
            print(f"Error loading PTB-XL dataset: {e}")
            return False
    
    def analyze_individual_datasets(self):
        """Analyze each dataset separately"""
        print("\n" + "="*60)
        print("📊 INDIVIDUAL DATASET ANALYSIS")
        print("="*60)
        
        datasets = []
        if self.original_data:
            datasets.append(('ORIGINAL', self.original_data))
        if self.new_data:
            datasets.append(('PTB-XL', self.new_data))
        
        for name, data in datasets:
            labels = data['labels']
            signals = data['signals']
            
            print(f"\n🏷️  {name} DATASET:")
            print(f"   Total samples: {len(labels):,}")
            print(f"   Signal shape: {signals.shape}")
            print(f"   Signal length: {signals.shape[1] if len(signals.shape) > 1 else 'N/A'}")
            
            # Class distribution
            class_counts = Counter(labels)
            print(f"   Classes found: {sorted(class_counts.keys())}")
            
            for class_name in sorted(class_counts.keys()):
                count = class_counts[class_name]
                percentage = (count / len(labels)) * 100
                print(f"      {class_name}: {count:,} samples ({percentage:.1f}%)")
    
    def combine_datasets(self):
        """Combine original + PTB-XL datasets"""
        if not self.original_data or not self.new_data:
            print("❌ Cannot combine - missing datasets")
            return False
        
        print("\n" + "="*60)
        print("🔄 COMBINING DATASETS")
        print("="*60)
        
        # Get data
        orig_signals = self.original_data['signals']
        orig_labels = self.original_data['labels']
        new_signals = self.new_data['signals']
        new_labels = self.new_data['labels']
        
        print(f"Original dataset: {orig_signals.shape}")
        print(f"PTB-XL dataset: {new_signals.shape}")
        
        # Check if signal lengths match
        if orig_signals.shape[1] != new_signals.shape[1]:
            print(f"⚠️  Signal length mismatch!")
            print(f"   Original: {orig_signals.shape[1]} samples")
            print(f"   PTB-XL: {new_signals.shape[1]} samples")
            
            # Standardize to shorter length
            min_length = min(orig_signals.shape[1], new_signals.shape[1])
            print(f"   Standardizing both to {min_length} samples")
            
            orig_signals = orig_signals[:, :min_length]
            new_signals = new_signals[:, :min_length]
        
        # Combine
        combined_signals = np.vstack([orig_signals, new_signals])
        combined_labels = np.hstack([orig_labels, new_labels])
        
        # Create source tracking
        sources = (['original'] * len(orig_labels) + 
                  ['ptb_xl'] * len(new_labels))
        
        self.combined_data = {
            'signals': combined_signals,
            'labels': combined_labels,
            'sources': sources
        }
        
        print(f"✅ Combined dataset shape: {combined_signals.shape}")
        return True
    
    def analyze_combined_dataset(self):
        """Analyze the combined dataset"""
        if not self.combined_data:
            print("❌ No combined dataset available")
            return
        
        print("\n" + "="*60)
        print("🎯 COMBINED DATASET ANALYSIS")
        print("="*60)
        
        labels = self.combined_data['labels']
        sources = self.combined_data['sources']
        
        total_samples = len(labels)
        print(f"🔢 Total combined samples: {total_samples:,}")
        
        # Overall class distribution
        class_counts = Counter(labels)
        print(f"\n📊 OVERALL CLASS DISTRIBUTION:")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            percentage = (count / total_samples) * 100
            print(f"   {class_name}: {count:,} samples ({percentage:.1f}%)")
        
        # By source breakdown
        print(f"\n📂 BY SOURCE BREAKDOWN:")
        source_counts = Counter(sources)
        for source, count in source_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   {source}: {count:,} samples ({percentage:.1f}%)")
        
        # Class distribution by source
        print(f"\n🔍 CLASS DISTRIBUTION BY SOURCE:")
        df = pd.DataFrame({'class': labels, 'source': sources})
        cross_tab = pd.crosstab(df['class'], df['source'], margins=True)
        print(cross_tab)
        
        # Calculate improvements
        print(f"\n🚀 IMPROVEMENT SUMMARY:")
        
        # Assume original had these approximate counts (adjust if you know exact numbers)
        original_estimates = {
            'F': 842,   # Your mentioned F class size
            'N': 2500,  # Estimated
            'Q': 500,   # Estimated  
            'S': 800,   # Estimated
            'V': 358    # Estimated
        }
        
        for class_name in sorted(class_counts.keys()):
            new_count = class_counts[class_name]
            old_count = original_estimates.get(class_name, 100)  # Default estimate
            improvement = new_count / old_count
            print(f"   Class {class_name}: {old_count:,} → {new_count:,} ({improvement:.1f}x improvement)")
        
        total_old = sum(original_estimates.values())
        total_improvement = total_samples / total_old
        print(f"   📈 Total dataset: {total_old:,} → {total_samples:,} ({total_improvement:.1f}x larger)")
    
    def plot_comprehensive_analysis(self):
        """Create comprehensive visualization"""
        if not all([self.original_data, self.new_data, self.combined_data]):
            print("❌ Missing data for comprehensive plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Dataset 1: Original
        orig_counts = Counter(self.original_data['labels'])
        axes[0,0].bar(orig_counts.keys(), orig_counts.values(), color='lightcoral', alpha=0.8)
        axes[0,0].set_title(f'Original Dataset\n({sum(orig_counts.values()):,} samples)')
        axes[0,0].set_ylabel('Count')
        
        # Dataset 2: PTB-XL
        new_counts = Counter(self.new_data['labels'])
        axes[0,1].bar(new_counts.keys(), new_counts.values(), color='lightblue', alpha=0.8)
        axes[0,1].set_title(f'PTB-XL Dataset\n({sum(new_counts.values()):,} samples)')
        
        # Dataset 3: Combined
        combined_counts = Counter(self.combined_data['labels'])
        axes[0,2].bar(combined_counts.keys(), combined_counts.values(), color='lightgreen', alpha=0.8)
        axes[0,2].set_title(f'Combined Dataset\n({sum(combined_counts.values()):,} samples)')
        
        # Class comparison across datasets
        all_classes = sorted(set(self.original_data['labels']) | set(self.new_data['labels']))
        
        orig_values = [orig_counts.get(c, 0) for c in all_classes]
        new_values = [new_counts.get(c, 0) for c in all_classes]
        combined_values = [combined_counts.get(c, 0) for c in all_classes]
        
        x = np.arange(len(all_classes))
        width = 0.25
        
        axes[1,0].bar(x - width, orig_values, width, label='Original', color='lightcoral', alpha=0.8)
        axes[1,0].bar(x, new_values, width, label='PTB-XL', color='lightblue', alpha=0.8)
        axes[1,0].bar(x + width, combined_values, width, label='Combined', color='lightgreen', alpha=0.8)
        
        axes[1,0].set_xlabel('Class')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Class Count Comparison')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(all_classes)
        axes[1,0].legend()
        axes[1,0].set_yscale('log')  # Log scale for better visualization
        
        # Improvement factors
        improvements = []
        class_names = []
        for c in all_classes:
            old_count = orig_counts.get(c, 1)  # Avoid division by zero
            new_total = combined_counts.get(c, 0)
            improvement = new_total / old_count
            improvements.append(improvement)
            class_names.append(c)
        
        axes[1,1].bar(class_names, improvements, color='gold', alpha=0.8)
        axes[1,1].set_xlabel('Class')
        axes[1,1].set_ylabel('Improvement Factor (x)')
        axes[1,1].set_title('Dataset Size Improvement by Class')
        axes[1,1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No improvement')
        
        # Sample signals comparison
        self.plot_sample_signals_comparison(axes[1,2])
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_sample_signals_comparison(self, ax):
        """Plot sample signals from different sources"""
        if not self.combined_data:
            return
        
        signals = self.combined_data['signals']
        labels = self.combined_data['labels']
        sources = self.combined_data['sources']
        
        # Find one signal from each source for comparison
        for i, source in enumerate(['original', 'ptb_xl']):
            mask = np.array(sources) == source
            if np.any(mask):
                source_signals = signals[mask]
                if len(source_signals) > 0:
                    sample_signal = source_signals[0]
                    ax.plot(sample_signal, alpha=0.7, 
                           label=f'{source.title()} (len={len(sample_signal)})',
                           linewidth=1)
        
        ax.set_title('Sample Signal Comparison')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def save_combined_dataset(self, output_dir="./combined_ecg_data"):
        """Save the final combined dataset"""
        if not self.combined_data:
            print("❌ No combined dataset to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save arrays
        np.save(output_path / 'X_combined.npy', self.combined_data['signals'])
        np.save(output_path / 'y_combined.npy', self.combined_data['labels'])
        np.save(output_path / 'sources_combined.npy', self.combined_data['sources'])
        
        # Save detailed metadata
        class_counts = Counter(self.combined_data['labels'])
        source_counts = Counter(self.combined_data['sources'])
        
        metadata = {
            'total_samples': len(self.combined_data['labels']),
            'signal_shape': list(self.combined_data['signals'].shape),
            'class_distribution': dict(class_counts),
            'source_distribution': dict(source_counts),
            'datasets_combined': list(source_counts.keys()),
            'creation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path / 'combined_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Combined dataset saved to: {output_path}")
        print(f"   Files: X_combined.npy, y_combined.npy, sources_combined.npy")
        
        return output_path
    
    def generate_summary_report(self):
        """Generate a comprehensive text summary"""
        print("\n" + "="*80)
        print("🎯 COMPLETE ECG DATASET SUMMARY REPORT")
        print("="*80)
        
        # Original dataset summary
        if self.original_data:
            orig_labels = self.original_data['labels']
            orig_counts = Counter(orig_labels)
            print(f"\n📂 ORIGINAL DATASET:")
            print(f"   Total samples: {len(orig_labels):,}")
            print(f"   Classes: {sorted(orig_counts.keys())}")
            print(f"   Most common: {orig_counts.most_common(1)[0][0]} ({orig_counts.most_common(1)[0][1]:,} samples)")
            print(f"   Least common: {orig_counts.most_common()[-1][0]} ({orig_counts.most_common()[-1][1]:,} samples)")
        else:
            print(f"\n📂 ORIGINAL DATASET: Not found")
        
        # PTB-XL dataset summary
        if self.new_data:
            new_labels = self.new_data['labels']
            new_counts = Counter(new_labels)
            print(f"\n📂 PTB-XL DATASET:")
            print(f"   Total samples: {len(new_labels):,}")
            print(f"   Classes: {sorted(new_counts.keys())}")
            print(f"   Most common: {new_counts.most_common(1)[0][0]} ({new_counts.most_common(1)[0][1]:,} samples)")
            print(f"   Least common: {new_counts.most_common()[-1][0]} ({new_counts.most_common()[-1][1]:,} samples)")
        else:
            print(f"\n📂 PTB-XL DATASET: Not processed yet")
        
        # Combined dataset summary
        if self.combined_data:
            combined_labels = self.combined_data['labels']
            combined_counts = Counter(combined_labels)
            
            print(f"\n🎯 COMBINED DATASET:")
            print(f"   Total samples: {len(combined_labels):,}")
            print(f"   Classes: {sorted(combined_counts.keys())}")
            
            print(f"\n📊 FINAL CLASS DISTRIBUTION:")
            for class_name in sorted(combined_counts.keys()):
                count = combined_counts[class_name]
                percentage = (count / len(combined_labels)) * 100
                print(f"      {class_name}: {count:,} samples ({percentage:.1f}%)")
            
            # Calculate improvements if original data exists
            if self.original_data:
                print(f"\n🚀 IMPROVEMENT ANALYSIS:")
                orig_total = len(self.original_data['labels'])
                combined_total = len(combined_labels)
                total_improvement = combined_total / orig_total
                
                print(f"   📈 Total samples: {orig_total:,} → {combined_total:,} ({total_improvement:.1f}x increase)")
                
                # Per-class improvements
                orig_counts = Counter(self.original_data['labels'])
                for class_name in sorted(combined_counts.keys()):
                    old_count = orig_counts.get(class_name, 0)
                    new_count = combined_counts[class_name]
                    
                    if old_count > 0:
                        improvement = new_count / old_count
                        added = new_count - old_count
                        print(f"   Class {class_name}: {old_count:,} → {new_count:,} (+{added:,}, {improvement:.1f}x)")
                    else:
                        print(f"   Class {class_name}: 0 → {new_count:,} (NEW CLASS!)")
            
            # Class balance analysis
            min_class_count = min(combined_counts.values())
            max_class_count = max(combined_counts.values())
            imbalance_ratio = max_class_count / min_class_count
            
            print(f"\n⚖️  CLASS BALANCE ANALYSIS:")
            print(f"   Most frequent class: {max_class_count:,} samples")
            print(f"   Least frequent class: {min_class_count:,} samples")
            print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
            
            if imbalance_ratio > 10:
                print(f"   ⚠️  HIGH IMBALANCE - Consider balancing techniques")
            elif imbalance_ratio > 5:
                print(f"   ⚡ MODERATE IMBALANCE - Manageable with class weights")
            else:
                print(f"   ✅ GOOD BALANCE - Ready for training!")
        
        print("\n" + "="*80)
    
    def run_complete_analysis(self, original_path="./data", ptb_path="./ecg_datasets/processed"):
        """Run complete analysis pipeline"""
        print("🔍 ECG Dataset Complete Analysis Starting...")
        
        # Try to load both datasets
        has_original = self.load_original_dataset(original_path)
        has_new = self.load_new_ptb_dataset(ptb_path)
        
        if not has_original and not has_new:
            print("❌ No datasets found! Make sure you have either:")
            print("   1. Original dataset in ./data/ folder")
            print("   2. PTB-XL processed data in ./ecg_datasets/processed/")
            return
        
        # Analyze individual datasets
        self.analyze_individual_datasets()
        
        # Combine if both exist
        if has_original and has_new:
            success = self.combine_datasets()
            if success:
                self.analyze_combined_dataset()
                
                # Create visualizations
                self.plot_comprehensive_analysis()
                
                # Save combined dataset
                output_path = self.save_combined_dataset()
                
                print(f"\n🎉 ANALYSIS COMPLETE!")
                print(f"💾 Combined dataset ready at: {output_path}")
                print(f"📊 Use X_combined.npy and y_combined.npy for training!")
        
        elif has_new:
            print("\n💡 Only PTB-XL data found. This is your new dataset!")
            # Treat PTB-XL as the main dataset
            self.combined_data = self.new_data.copy()
            self.combined_data['sources'] = ['ptb_xl'] * len(self.new_data['labels'])
            self.analyze_combined_dataset()
        
        elif has_original:
            print("\n💡 Only original dataset found. Run PTB-XL processing first!")

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive ECG Dataset Analysis')
    parser.add_argument('--original-path', default='./data', 
                       help='Path to original dataset')
    parser.add_argument('--ptb-path', default='./ecg_datasets/processed',
                       help='Path to PTB-XL processed data') 
    parser.add_argument('--save-combined', action='store_true',
                       help='Save combined dataset')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ECGDatasetAnalyzer()
    analyzer.run_complete_analysis(args.original_path, args.ptb_path)

if __name__ == "__main__":
    main()