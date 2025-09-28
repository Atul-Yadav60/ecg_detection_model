#!/usr/bin/env python3
"""
Enhanced ECG Dataset Combiner
Combines your original processed dataset (train/val/test) with PTB-XL data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter
import os

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
class EnhancedECGCombiner:
    def __init__(self):
        self.original_data = None
        self.ptb_data = None
        self.combined_data = None
        
    def load_original_dataset(self, data_path="./data/processed"):
        """Load your original train/val/test dataset"""
        try:
            data_dir = Path(data_path)
            
            if not data_dir.exists():
                print(f"❌ Original dataset path not found: {data_path}")
                return False
            
            # Load train/val/test splits
            splits = ['train', 'val', 'test']
            all_signals = []
            all_labels = []
            split_info = {}
            
            for split in splits:
                split_dir = data_dir / split
                segments_file = split_dir / 'segments.npy'
                labels_file = split_dir / 'labels.npy'
                
                if segments_file.exists() and labels_file.exists():
                    segments = np.load(segments_file)
                    labels = np.load(labels_file)
                    
                    all_signals.append(segments)
                    all_labels.append(labels)
                    split_info[split] = {
                        'samples': len(labels),
                        'shape': segments.shape
                    }
                    
                    print(f"📂 Loaded {split}: {segments.shape} signals, {len(labels)} labels")
                else:
                    print(f"⚠️  Missing {split} data files")
            
            if all_signals:
                # Combine all splits
                combined_signals = np.vstack(all_signals)
                combined_labels = np.hstack(all_labels)
                
                # Load label encoding
                label_encoding_file = data_dir / 'label_encoding.json'
                label_map = {}
                if label_encoding_file.exists():
                    with open(label_encoding_file, 'r') as f:
                        label_map = json.load(f)
                
                self.original_data = {
                    'signals': combined_signals,
                    'labels': combined_labels,
                    'source': 'Original Dataset',
                    'split_info': split_info,
                    'label_map': label_map,
                    'total_samples': len(combined_labels)
                }
                
                print(f"\n✅ Original dataset loaded successfully:")
                print(f"   Total samples: {len(combined_labels):,}")
                print(f"   Signal shape: {combined_signals.shape}")
                print(f"   Splits loaded: {list(split_info.keys())}")
                
                return True
            else:
                print("❌ No data files found in splits")
                return False
                
        except Exception as e:
            print(f"Error loading original dataset: {e}")
            return False
    
    def load_ptb_dataset(self, data_path="./ecg_datasets/processed"):
        """Load your PTB-XL processed dataset"""
        try:
            data_dir = Path(data_path)
            
            if not data_dir.exists():
                print(f"❌ PTB-XL data not found at {data_path}")
                return False
            
            # Load PTB-XL processed data
            signals_file = data_dir / 'combined_ecg_signals.npy'
            labels_file = data_dir / 'combined_ecg_labels.npy'
            
            if not signals_file.exists() or not labels_file.exists():
                print(f"❌ PTB-XL processed files not found")
                return False
            
            signals = np.load(signals_file)
            labels = np.load(labels_file)
            
            # Load metadata if available
            metadata_file = data_dir / 'metadata.json'
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            self.ptb_data = {
                'signals': signals,
                'labels': labels,
                'source': 'PTB-XL Dataset',
                'metadata': metadata,
                'total_samples': len(labels)
            }
            
            print(f"\n✅ PTB-XL dataset loaded successfully:")
            print(f"   Total samples: {len(labels):,}")
            print(f"   Signal shape: {signals.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading PTB-XL dataset: {e}")
            return False
    
    def analyze_original_detailed(self):
        """Detailed analysis of original dataset"""
        if not self.original_data:
            return
        
        print("\n" + "="*60)
        print("📊 ORIGINAL DATASET DETAILED ANALYSIS")
        print("="*60)
        
        labels = self.original_data['labels']
        signals = self.original_data['signals']
        split_info = self.original_data['split_info']
        
        # Convert numeric labels to class names if needed
        label_map = self.original_data.get('label_map', {})
        if label_map and isinstance(labels[0], (int, np.integer)):
            # Reverse the label map to get class names
            reverse_map = {v: k for k, v in label_map.items()}
            class_labels = [reverse_map.get(label, f'Class_{label}') for label in labels]
        else:
            class_labels = labels
        
        # Overall stats
        total_samples = len(labels)
        print(f"🔢 Total samples across all splits: {total_samples:,}")
        print(f"📏 Signal dimensions: {signals.shape}")
        print(f"🕒 Signal length: {signals.shape[1]} samples")
        
        # Split breakdown
        print(f"\n📂 SPLIT BREAKDOWN:")
        for split, info in split_info.items():
            count = info['samples']
            percentage = (count / total_samples) * 100
            print(f"   {split}: {count:,} samples ({percentage:.1f}%)")
        
        # Class distribution
        class_counts = Counter(class_labels)
        print(f"\n🏷️  CLASS DISTRIBUTION:")
        print(f"   Total classes: {len(class_counts)}")
        
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            percentage = (count / total_samples) * 100
            print(f"      {class_name}: {count:,} samples ({percentage:.1f}%)")
        
        # Verify your provided counts
        expected_counts = {
            'F': 842, 'N': 122474, 'Q': 16026, 'S': 4658, 'V': 10540
        }
        
        print(f"\n🔍 VERIFICATION vs PROVIDED COUNTS:")
        for class_name, expected in expected_counts.items():
            actual = class_counts.get(class_name, 0)
            if actual == expected:
                status = "✅ MATCH"
            else:
                status = f"❓ DIFF (expected {expected:,})"
            print(f"   {class_name}: {actual:,} {status}")
    
    def analyze_datasets_separately(self):
        """Analyze both datasets before combining"""
        print("\n" + "="*80)
        print("📊 SEPARATE DATASET ANALYSIS")
        print("="*80)
        
        datasets = []
        if self.original_data:
            datasets.append(('ORIGINAL', self.original_data))
        if self.ptb_data:
            datasets.append(('PTB-XL', self.ptb_data))
        
        for name, data in datasets:
            print(f"\n🔍 {name} DATASET:")
            
            labels = data['labels']
            signals = data['signals']
            
            # Handle different label types
            if isinstance(labels[0], (int, np.integer)):
                # Numeric labels - convert using label map if available
                label_map = data.get('label_map', {})
                if label_map:
                    reverse_map = {v: k for k, v in label_map.items()}
                    display_labels = [reverse_map.get(label, f'Class_{label}') for label in labels]
                else:
                    display_labels = [str(label) for label in labels]
            else:
                # String labels
                display_labels = [str(label) for label in labels]
            
            print(f"   Total samples: {len(labels):,}")
            print(f"   Signal shape: {signals.shape}")
            print(f"   Data type: {signals.dtype}")
            
            # Class distribution
            class_counts = Counter(display_labels)
            print(f"   Classes: {sorted(class_counts.keys())}")
            
            for class_name in sorted(class_counts.keys()):
                count = class_counts[class_name]
                percentage = (count / len(labels)) * 100
                print(f"      {class_name}: {count:,} samples ({percentage:.1f}%)")
    
    def standardize_labels(self):
        """Standardize label formats for combining"""
        print("\n🔄 STANDARDIZING LABELS...")
        
        # Standardize original labels
        if self.original_data:
            orig_labels = self.original_data['labels']
            if isinstance(orig_labels[0], (int, np.integer)):
                # Convert numeric to string using label map
                label_map = self.original_data.get('label_map', {})
                if label_map:
                    reverse_map = {v: k for k, v in label_map.items()}
                    standardized_orig = [reverse_map.get(label, f'Class_{label}') for label in orig_labels]
                else:
                    standardized_orig = [str(label) for label in orig_labels]
            else:
                standardized_orig = [str(label) for label in orig_labels]
            
            self.original_data['standardized_labels'] = standardized_orig
            print(f"   Original labels standardized: {Counter(standardized_orig)}")
        
        # Standardize PTB-XL labels
        if self.ptb_data:
            ptb_labels = self.ptb_data['labels']
            standardized_ptb = [str(label) for label in ptb_labels]
            self.ptb_data['standardized_labels'] = standardized_ptb
            print(f"   PTB-XL labels standardized: {Counter(standardized_ptb)}")
    
    def combine_datasets(self):
        """Combine original + PTB-XL datasets with proper handling"""
        if not self.original_data or not self.ptb_data:
            print("❌ Cannot combine - missing datasets")
            return False
        
        print("\n" + "="*60)
        print("🔄 COMBINING DATASETS")
        print("="*60)
        
        # Standardize labels first
        self.standardize_labels()
        
        # Get standardized data
        orig_signals = self.original_data['signals']
        orig_labels = self.original_data['standardized_labels']
        ptb_signals = self.ptb_data['signals']
        ptb_labels = self.ptb_data['standardized_labels']
        
        print(f"Original dataset: {orig_signals.shape}")
        print(f"PTB-XL dataset: {ptb_signals.shape}")
        
        # Check signal compatibility
        if orig_signals.shape[1] != ptb_signals.shape[1]:
            print(f"⚠️  Signal length mismatch!")
            print(f"   Original: {orig_signals.shape[1]} samples")
            print(f"   PTB-XL: {ptb_signals.shape[1]} samples")
            
            # Standardize to shorter length
            min_length = min(orig_signals.shape[1], ptb_signals.shape[1])
            print(f"   Truncating both to {min_length} samples")
            
            orig_signals = orig_signals[:, :min_length]
            ptb_signals = ptb_signals[:, :min_length]
        
        # Combine datasets
        combined_signals = np.vstack([orig_signals, ptb_signals])
        combined_labels = np.array(orig_labels + ptb_labels)
        
        # Create source tracking
        sources = (['original'] * len(orig_labels) + 
                  ['ptb_xl'] * len(ptb_labels))
        
        self.combined_data = {
            'signals': combined_signals,
            'labels': combined_labels,
            'sources': sources,
            'total_samples': len(combined_labels)
        }
        
        print(f"✅ Successfully combined datasets!")
        print(f"   Final shape: {combined_signals.shape}")
        print(f"   Total samples: {len(combined_labels):,}")
        
        return True
    
    def analyze_combined_comprehensive(self):
        """Comprehensive analysis of combined dataset"""
        if not self.combined_data:
            print("❌ No combined dataset available")
            return
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE COMBINED DATASET ANALYSIS")
        print("="*80)
        
        labels = self.combined_data['labels']
        sources = self.combined_data['sources']
        signals = self.combined_data['signals']
        
        total_samples = len(labels)
        print(f"🔢 TOTAL COMBINED SAMPLES: {total_samples:,}")
        print(f"📏 Final signal dimensions: {signals.shape}")
        
        # Overall class distribution
        class_counts = Counter(labels)
        print(f"\n📊 FINAL CLASS DISTRIBUTION:")
        print(f"   Total classes: {len(class_counts)}")
        
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            percentage = (count / total_samples) * 100
            print(f"      {class_name}: {count:,} samples ({percentage:.1f}%)")
        
        # Source breakdown
        source_counts = Counter(sources)
        print(f"\n📂 CONTRIBUTION BY SOURCE:")
        for source, count in source_counts.items():
            percentage = (count / total_samples) * 100
            print(f"   {source}: {count:,} samples ({percentage:.1f}%)")
        
        # Cross-tabulation: Class vs Source
        print(f"\n🔍 CLASS DISTRIBUTION BY SOURCE:")
        df = pd.DataFrame({'class': labels, 'source': sources})
        cross_tab = pd.crosstab(df['class'], df['source'], margins=True)
        print(cross_tab)
        
        # Calculate precise improvements based on your provided original counts
        original_counts = {
            'F': 842, 'N': 122474, 'Q': 16026, 'S': 4658, 'V': 10540
        }
        
        print(f"\n🚀 DETAILED IMPROVEMENT ANALYSIS:")
        print(f"{'Class':<5} {'Original':<10} {'PTB-XL':<10} {'Combined':<10} {'Improvement':<12} {'Added':<10}")
        print("-" * 70)
        
        total_orig = sum(original_counts.values())
        total_ptb_added = 0
        
        for class_name in sorted(class_counts.keys()):
            orig_count = original_counts.get(class_name, 0)
            
            # Count PTB-XL contributions for this class
            ptb_mask = (np.array(sources) == 'ptb_xl') & (np.array(labels) == class_name)
            ptb_count = np.sum(ptb_mask)
            total_ptb_added += ptb_count
            
            combined_count = class_counts[class_name]
            
            if orig_count > 0:
                improvement = combined_count / orig_count
                improvement_str = f"{improvement:.1f}x"
            else:
                improvement_str = "NEW!"
            
            print(f"{class_name:<5} {orig_count:<10,} {ptb_count:<10,} {combined_count:<10,} "
                  f"{improvement_str:<12} {ptb_count:<10,}")
        
        # Overall summary
        total_improvement = total_samples / total_orig
        print("-" * 70)
        print(f"TOTAL {total_orig:<10,} {total_ptb_added:<10,} {total_samples:<10,} "
              f"{total_improvement:.1f}x{'':<7} {total_ptb_added:<10,}")
        
        # Balance analysis
        print(f"\n⚖️  CLASS BALANCE ANALYSIS:")
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count
        
        print(f"   Most frequent: {max_count:,} samples")
        print(f"   Least frequent: {min_count:,} samples")
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 50:
            print(f"   🔴 SEVERE IMBALANCE - Strong balancing needed")
        elif imbalance_ratio > 20:
            print(f"   🟡 HIGH IMBALANCE - Consider SMOTE or class weights")
        elif imbalance_ratio > 5:
            print(f"   🟠 MODERATE IMBALANCE - Use class weights")
        else:
            print(f"   🟢 GOOD BALANCE")
    
    def create_comprehensive_plots(self):
        """Create detailed visualizations"""
        if not all([self.original_data, self.ptb_data, self.combined_data]):
            print("❌ Missing data for comprehensive plotting")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Class distribution comparison (before/after)
        ax1 = plt.subplot(3, 3, 1)
        original_counts = Counter(self.original_data['standardized_labels'])
        ax1.bar(original_counts.keys(), original_counts.values(), 
                color='lightcoral', alpha=0.8)
        ax1.set_title(f'Original Dataset\n({sum(original_counts.values()):,} samples)')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. PTB-XL contribution
        ax2 = plt.subplot(3, 3, 2)
        ptb_counts = Counter(self.ptb_data['standardized_labels'])
        ax2.bar(ptb_counts.keys(), ptb_counts.values(), 
                color='lightblue', alpha=0.8)
        ax2.set_title(f'PTB-XL Addition\n({sum(ptb_counts.values()):,} samples)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Final combined
        ax3 = plt.subplot(3, 3, 3)
        combined_counts = Counter(self.combined_data['labels'])
        ax3.bar(combined_counts.keys(), combined_counts.values(), 
                color='lightgreen', alpha=0.8)
        ax3.set_title(f'Combined Dataset\n({sum(combined_counts.values()):,} samples)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Side-by-side comparison
        ax4 = plt.subplot(3, 3, 4)
        all_classes = sorted(set(original_counts.keys()) | set(combined_counts.keys()))
        
        x = np.arange(len(all_classes))
        width = 0.35
        
        orig_values = [original_counts.get(c, 0) for c in all_classes]
        combined_values = [combined_counts.get(c, 0) for c in all_classes]
        
        ax4.bar(x - width/2, orig_values, width, label='Original', 
                color='lightcoral', alpha=0.8)
        ax4.bar(x + width/2, combined_values, width, label='Combined', 
                color='lightgreen', alpha=0.8)
        
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Count')
        ax4.set_title('Before vs After Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(all_classes)
        ax4.legend()
        ax4.set_yscale('log')
        
        # 5. Improvement factors
        ax5 = plt.subplot(3, 3, 5)
        improvements = []
        for c in all_classes:
            old_count = original_counts.get(c, 1)
            new_count = combined_counts.get(c, 0)
            improvement = new_count / old_count
            improvements.append(improvement)
        
        bars = ax5.bar(all_classes, improvements, color='gold', alpha=0.8)
        ax5.set_xlabel('Class')
        ax5.set_ylabel('Improvement Factor')
        ax5.set_title('Dataset Size Improvement by Class')
        ax5.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{improvement:.1f}x', ha='center', va='bottom')
        
        # 6. Sample signals comparison
        ax6 = plt.subplot(3, 3, 6)
        
        # Plot sample signals from each dataset
        orig_signal = self.original_data['signals'][0]
        ptb_signal = self.ptb_data['signals'][0]
        
        ax6.plot(orig_signal[:500], label='Original Sample', alpha=0.7, linewidth=1)
        ax6.plot(ptb_signal[:500], label='PTB-XL Sample', alpha=0.7, linewidth=1)
        ax6.set_title('Sample Signal Comparison')
        ax6.set_xlabel('Time (samples)')
        ax6.set_ylabel('Amplitude')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Class balance heatmap
        ax7 = plt.subplot(3, 3, 7)
        
        # Create percentage matrix for heatmap
        df_source = pd.DataFrame({'class': self.combined_data['labels'], 
                                 'source': self.combined_data['sources']})
        
        # Calculate percentages within each class
        cross_pct = pd.crosstab(df_source['class'], df_source['source'], normalize='index') * 100
        
        sns.heatmap(cross_pct, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   ax=ax7, cbar_kws={'label': 'Percentage'})
        ax7.set_title('Source Contribution by Class (%)')
        ax7.set_xlabel('Data Source')
        ax7.set_ylabel('Class')
        
        # 8. Dataset size evolution
        ax8 = plt.subplot(3, 3, 8)
        
        categories = ['Original', 'PTB-XL\nAddition', 'Combined\nTotal']
        values = [
            sum(original_counts.values()),
            sum(ptb_counts.values()),
            sum(combined_counts.values())
        ]
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        bars = ax8.bar(categories, values, color=colors, alpha=0.8)
        
        ax8.set_ylabel('Sample Count')
        ax8.set_title('Dataset Growth Summary')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,}', ha='center', va='bottom')
        
        # 9. Class distribution pie chart
        ax9 = plt.subplot(3, 3, 9)
        
        class_names = list(combined_counts.keys())
        class_values = list(combined_counts.values())
        
        # Create pie chart with better colors
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        
        wedges, texts, autotexts = ax9.pie(class_values, labels=class_names, 
                                          autopct='%1.1f%%', colors=colors_pie)
        ax9.set_title('Final Combined Class Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_combined_dataset(self, output_dir="./combined_ecg_final"):
        """Save the final combined dataset with comprehensive metadata"""
        if not self.combined_data:
            print("❌ No combined dataset to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        signals = self.combined_data['signals']
        labels = self.combined_data['labels']
        sources = self.combined_data['sources']
        
        # Save main arrays
        np.save(output_path / 'X_final_combined.npy', signals)
        np.save(output_path / 'y_final_combined.npy', labels)
        np.save(output_path / 'sources_final_combined.npy', sources)
        
        # Create detailed metadata
        class_counts = Counter(labels)
        source_counts = Counter(sources)
        
        # Calculate improvements
        original_totals = {
            'F': 842, 'N': 122474, 'Q': 16026, 'S': 4658, 'V': 10540
        }
        
        improvements = {}
        ptb_contributions = {}
        
        for class_name in class_counts.keys():
            orig_count = original_totals.get(class_name, 0)
            final_count = class_counts[class_name]
            
            # Calculate PTB-XL contribution
            ptb_mask = (np.array(sources) == 'ptb_xl') & (np.array(labels) == class_name)
            ptb_contrib = np.sum(ptb_mask)
            
            if orig_count > 0:
                improvement = final_count / orig_count
            else:
                improvement = float('inf')  # New class
            
            improvements[class_name] = improvement
            ptb_contributions[class_name] =int(ptb_contrib)
        
        metadata = {
            'creation_info': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_samples': int(len(labels)),
                'signal_shape': [int(x) for x in signals.shape],
                'signal_length': int(signals.shape[1]),
                'total_classes': len(class_counts)
            },
            'original_dataset': {
                'total_samples': sum(original_totals.values()),
                'class_distribution': original_totals,
                'source_splits': ['train', 'val', 'test'] if self.original_data else []
            },
            'ptb_xl_dataset': {
                'total_samples': int(source_counts.get('ptb_xl', 0)),
                'contribution_by_class': ptb_contributions
            },
            'combined_dataset': {
                'total_samples': int(len(labels)),
                'final_class_distribution': {k: int(v) for k, v in class_counts.items()},
                'source_distribution': {k: int(v) for k, v in source_counts.items()},
                'improvement_factors': {k: float(v) for k, v in improvements.items()},
                'total_improvement_factor': float(len(labels) / sum(original_totals.values()))
            },
            'balance_metrics': {
                'most_frequent_class': max(class_counts, key=class_counts.get),
                'most_frequent_count': int(max(class_counts.values())),
                'least_frequent_class': min(class_counts, key=class_counts.get),
                'least_frequent_count': int(min(class_counts.values())),
                'imbalance_ratio': float(max(class_counts.values()) / min(class_counts.values()))
            }
        }
        metadata = convert_numpy_types(metadata)
        # Save metadata
        with open(output_path / 'final_combined_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save summary report
        with open(output_path / 'dataset_summary_report.txt', 'w') as f:
            f.write("ECG DATASET COMBINATION SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"FINAL COMBINED DATASET:\n")
            f.write(f"Total samples: {len(labels):,}\n")
            f.write(f"Signal dimensions: {signals.shape}\n")
            f.write(f"Classes: {len(class_counts)}\n\n")
            
            f.write("CLASS DISTRIBUTION:\n")
            for class_name, count in sorted(class_counts.items()):
                pct = (count / len(labels)) * 100
                f.write(f"  {class_name}: {count:,} ({pct:.1f}%)\n")
            
            f.write(f"\nIMPROVEMENT SUMMARY:\n")
            f.write(f"Original total: {sum(original_totals.values()):,}\n")
            f.write(f"PTB-XL added: {source_counts.get('ptb_xl', 0):,}\n")
            f.write(f"Final total: {len(labels):,}\n")
            f.write(f"Overall improvement: {len(labels) / sum(original_totals.values()):.1f}x\n")
        
        print(f"\n💾 Complete combined dataset saved to: {output_path}")
        print(f"📁 Files created:")
        print(f"   - X_final_combined.npy (signals)")
        print(f"   - y_final_combined.npy (labels)")  
        print(f"   - sources_final_combined.npy (source tracking)")
        print(f"   - final_combined_metadata.json (complete metadata)")
        print(f"   - dataset_summary_report.txt (text summary)")
        
        return output_path
    
    def create_train_val_test_splits(self, output_dir="./combined_ecg_final", 
                                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Create new train/val/test splits from combined data"""
        if not self.combined_data:
            print("❌ No combined dataset to split")
            return
        
        print(f"\n🔄 CREATING NEW TRAIN/VAL/TEST SPLITS")
        print(f"   Ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
        
        signals = self.combined_data['signals']
        labels = self.combined_data['labels']
        sources = self.combined_data['sources']
        
        # Stratified split to maintain class proportions
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val+test)
        X_train, X_temp, y_train, y_temp, src_train, src_temp = train_test_split(
            signals, labels, sources, 
            test_size=(val_ratio + test_ratio),
            random_state=42,
            stratify=labels
        )
        
        # Second split: val vs test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test, src_val, src_test = train_test_split(
            X_temp, y_temp, src_temp,
            test_size=val_test_ratio,
            random_state=42,
            stratify=y_temp
        )
        
        # Save splits
        output_path = Path(output_dir)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (output_path / split).mkdir(exist_ok=True)
        
        # Save train
        np.save(output_path / 'train' / 'segments.npy', X_train)
        np.save(output_path / 'train' / 'labels.npy', y_train)
        np.save(output_path / 'train' / 'sources.npy', src_train)
        
        # Save val
        np.save(output_path / 'val' / 'segments.npy', X_val)
        np.save(output_path / 'val' / 'labels.npy', y_val)
        np.save(output_path / 'val' / 'sources.npy', src_val)
        
        # Save test
        np.save(output_path / 'test' / 'segments.npy', X_test)
        np.save(output_path / 'test' / 'labels.npy', y_test)
        np.save(output_path / 'test' / 'sources.npy', src_test)
        
        # Print split summary
        print(f"\n📊 SPLIT SUMMARY:")
        split_data = [
            ('Train', X_train, y_train),
            ('Val', X_val, y_val), 
            ('Test', X_test, y_test)
        ]
        
        for split_name, X_split, y_split in split_data:
            split_counts = Counter(y_split)
            print(f"   {split_name}: {len(y_split):,} samples")
            for class_name in sorted(split_counts.keys()):
                count = split_counts[class_name]
                pct = (count / len(y_split)) * 100
                print(f"      {class_name}: {count:,} ({pct:.1f}%)")
        
        print(f"\n💾 New splits saved to: {output_path}")
        
        return {
            'train': (X_train, y_train, src_train),
            'val': (X_val, y_val, src_val),
            'test': (X_test, y_test, src_test)
        }
    
    def run_complete_pipeline(self, original_path="./data/processed", 
                            ptb_path="./ecg_datasets/processed"):
        """Run the complete combining and analysis pipeline"""
        print("🚀 ENHANCED ECG DATASET COMBINER STARTING...")
        print("="*80)
        
        # Load datasets
        print("\n1️⃣  LOADING DATASETS...")
        has_original = self.load_original_dataset(original_path)
        has_ptb = self.load_ptb_dataset(ptb_path)
        
        if not has_original:
            print(f"\n❌ Could not find original dataset at {original_path}")
            print("Expected structure:")
            print("  data/processed/train/{segments.npy, labels.npy}")
            print("  data/processed/val/{segments.npy, labels.npy}")
            print("  data/processed/test/{segments.npy, labels.npy}")
            
            if not has_ptb:
                return
            else:
                print("Continuing with PTB-XL data only...")
        
        if not has_ptb:
            print(f"\n❌ Could not find PTB-XL dataset at {ptb_path}")
            print("Expected files: combined_ecg_signals.npy, combined_ecg_labels.npy")
            
            if has_original:
                print("Continuing with original data only...")
                self.analyze_original_detailed()
            return
        
        # Analyze individual datasets
        print("\n2️⃣  ANALYZING INDIVIDUAL DATASETS...")
        if has_original:
            self.analyze_original_detailed()
        self.analyze_datasets_separately()
        
        # Combine datasets if both exist
        if has_original and has_ptb:
            print("\n3️⃣  COMBINING DATASETS...")
            success = self.combine_datasets()
            
            if success:
                print("\n4️⃣  COMPREHENSIVE ANALYSIS...")
                self.analyze_combined_comprehensive()
                
                print("\n5️⃣  CREATING VISUALIZATIONS...")
                self.create_comprehensive_plots()
                
                print("\n6️⃣  SAVING COMBINED DATASET...")
                output_path = self.save_combined_dataset()
                
                print("\n7️⃣  CREATING NEW TRAIN/VAL/TEST SPLITS...")
                splits = self.create_train_val_test_splits()
                
                total_samples = self.combined_data['total_samples']
                print(f"\n🎉 PIPELINE COMPLETE!")
                print(f"🎯 YOUR ENHANCED DATASET IS READY!")
                print(f"📂 Location: {output_path}")
                print(f"📊 Total samples: {total_samples:,}")
                print(f"🚀 Improvement: {total_samples / 154540:.1f}x larger!")
                
                return True
        
        elif has_ptb:
            print("\n💡 Only PTB-XL data available - treating as main dataset")
            # Use PTB data as combined data
            self.combined_data = {
                'signals': self.ptb_data['signals'],
                'labels': self.ptb_data['standardized_labels'],
                'sources': ['ptb_xl'] * len(self.ptb_data['labels']),
                'total_samples': len(self.ptb_data['labels'])
            }
            
            # Use PTB data as combined data
            self.combined_data = {
                'signals': self.ptb_data['signals'],
                'labels': self.ptb_data['standardized_labels'],
                'sources': ['ptb_xl'] * len(self.ptb_data['labels']),
                'total_samples': len(self.ptb_data['labels'])
            }
            
            self.analyze_combined_comprehensive()
            self.save_combined_dataset()
            self.create_train_val_test_splits()
            
        return False

def analyze_existing_datasets():
    """Quick function to analyze what datasets you currently have"""
    print("🔍 SCANNING FOR EXISTING DATASETS...")
    print("="*50)
    
    # Check original dataset location
    original_path = Path("./data/processed")
    print(f"\n📂 Original Dataset ({original_path}):")
    if original_path.exists():
        for split in ['train', 'val', 'test']:
            split_dir = original_path / split
            if split_dir.exists():
                segments_file = split_dir / 'segments.npy'
                labels_file = split_dir / 'labels.npy'
                
                if segments_file.exists() and labels_file.exists():
                    segments = np.load(segments_file)
                    labels = np.load(labels_file)
                    print(f"   ✅ {split}: {segments.shape} ({len(labels):,} samples)")
                else:
                    print(f"   ❌ {split}: Missing files")
            else:
                print(f"   ❌ {split}: Directory not found")
    else:
        print("   ❌ Original dataset directory not found")
    
    # Check PTB-XL dataset
    ptb_path = Path("./ecg_datasets/processed")
    print(f"\n📂 PTB-XL Dataset ({ptb_path}):")
    if ptb_path.exists():
        signals_file = ptb_path / 'combined_ecg_signals.npy'
        labels_file = ptb_path / 'combined_ecg_labels.npy'
        
        if signals_file.exists() and labels_file.exists():
            signals = np.load(signals_file)
            labels = np.load(labels_file)
            print(f"   ✅ PTB-XL: {signals.shape} ({len(labels):,} samples)")
        else:
            print(f"   ❌ PTB-XL: Missing processed files")
    else:
        print("   ❌ PTB-XL directory not found")
    
    print("\n" + "="*50)

def main():
    """Main execution function with enhanced options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced ECG Dataset Combiner')
    parser.add_argument('--original-path', default='./data/processed',
                       help='Path to original dataset (default: ./data/processed)')
    parser.add_argument('--ptb-path', default='./ecg_datasets/processed',
                       help='Path to PTB-XL processed data')
    parser.add_argument('--output-dir', default='./combined_ecg_final',
                       help='Output directory for combined dataset')
    parser.add_argument('--scan-only', action='store_true',
                       help='Only scan for existing datasets')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        print("❌ Train/val/test ratios must sum to 1.0")
        return
    
    if args.scan_only:
        analyze_existing_datasets()
        return
    
    # Run full pipeline
    combiner = EnhancedECGCombiner()
    success = combiner.run_complete_pipeline(
        original_path=args.original_path,
        ptb_path=args.ptb_path
    )
    
    if success:
        print(f"\n🎯 SUCCESS! Your enhanced ECG dataset is ready for training!")
        print(f"📂 Use files from: {args.output_dir}")
        print(f"🤖 Recommended next steps:")
        print(f"   1. Load X_final_combined.npy and y_final_combined.npy")
        print(f"   2. Or use the new train/val/test splits")
        print(f"   3. Consider class balancing due to imbalance")
        print(f"   4. Start training your improved model!")

if __name__ == "__main__":
    # You can also run this directly for quick analysis
    if len(os.sys.argv) == 1:  # No command line args
        print("🔍 Running quick dataset scan...")
        analyze_existing_datasets()
        
        print("\n" + "="*50)
        print("To run full combination, use:")
        print("python enhanced_ecg_combiner.py")
        print("Or with custom paths:")
        print("python enhanced_ecg_combiner.py --original-path ./data/processed --ptb-path ./ecg_datasets/processed")
    else:
        main()