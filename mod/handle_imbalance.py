#!/usr/bin/env python3
"""
Handle ECG class imbalance using various techniques
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_class_distribution(labels):
    """Analyze class distribution and identify imbalance"""
    
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    print(f"\n{'='*60}")
    print(f"📊 CLASS DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\n{'Class':<4} {'Count':<8} {'Percentage':<12} {'Status':<15}")
    print("-" * 50)
    
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total_samples) * 100
        
        if percentage > 50:
            status = "🚨 MAJORITY"
        elif percentage < 1:
            status = "🚨 SEVERE MINORITY"
        elif percentage < 5:
            status = "⚠️  MINORITY"
        else:
            status = "✅ BALANCED"
            
        print(f"{class_name:<4} {count:<8,} {percentage:<12.1f}% {status:<15}")
    
    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n📈 IMBALANCE METRICS:")
    print(f"   Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print(f"   Majority Class: {max_count:,} samples")
    print(f"   Minority Class: {min_count:,} samples")
    
    if imbalance_ratio > 100:
        print(f"   🚨 SEVERE IMBALANCE - Requires immediate attention!")
    elif imbalance_ratio > 10:
        print(f"   ⚠️  HIGH IMBALANCE - Needs balancing techniques")
    else:
        print(f"   ✅ ACCEPTABLE IMBALANCE")
    
    return class_counts, imbalance_ratio

def compute_class_weights(labels, method='balanced'):
    """Compute class weights for weighted loss"""
    
    print(f"\n{'='*60}")
    print(f"⚖️  COMPUTING CLASS WEIGHTS")
    print(f"{'='*60}")
    
    # Get unique classes and their counts
    unique_classes = np.unique(labels)
    class_counts = Counter(labels)
    
    if method == 'balanced':
        # Use sklearn's balanced weights
        weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=labels
        )
    elif method == 'inverse':
        # Inverse frequency weighting
        total_samples = len(labels)
        weights = [total_samples / (len(unique_classes) * class_counts[c]) for c in unique_classes]
    else:
        # Equal weights
        weights = [1.0] * len(unique_classes)
    
    # Create class weight dictionary
    class_weights = {cls: weight for cls, weight in zip(unique_classes, weights)}
    
    print(f"\n{'Class':<4} {'Count':<8} {'Weight':<10} {'Description':<20}")
    print("-" * 50)
    
    for cls in sorted(unique_classes):
        count = class_counts[cls]
        weight = class_weights[cls]
        
        if weight > 5:
            desc = "🚨 Very High Weight"
        elif weight > 2:
            desc = "⚠️  High Weight"
        else:
            desc = "✅ Normal Weight"
            
        print(f"{cls:<4} {count:<8,} {weight:<10.2f} {desc:<20}")
    
    return class_weights

def create_weighted_loss_function(class_weights, device='cuda'):
    """Create weighted CrossEntropyLoss"""
    
    # Convert to tensor and move to device
    weights_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], 
                                 dtype=torch.float32, device=device)
    
    print(f"\n⚖️  Weighted Loss Function Created:")
    print(f"   Weights tensor shape: {weights_tensor.shape}")
    print(f"   Device: {device}")
    print(f"   Weight range: {weights_tensor.min():.2f} - {weights_tensor.max():.2f}")
    
    return nn.CrossEntropyLoss(weight=weights_tensor)

def suggest_balancing_strategies(imbalance_ratio, class_counts):
    """Suggest appropriate balancing strategies"""
    
    print(f"\n{'='*60}")
    print(f"💡 RECOMMENDED BALANCING STRATEGIES")
    print(f"{'='*60}")
    
    strategies = []
    
    if imbalance_ratio > 100:
        print(f"🚨 SEVERE IMBALANCE DETECTED ({imbalance_ratio:.1f}:1)")
        strategies.extend([
            "1. 🎯 WEIGHTED LOSS FUNCTION (Immediate fix)",
            "2. 📈 DATA AUGMENTATION for minority classes",
            "3. ⬇️  UNDERSAMPLING majority class",
            "4. 🔄 COMBINE multiple techniques"
        ])
    elif imbalance_ratio > 10:
        print(f"⚠️  HIGH IMBALANCE DETECTED ({imbalance_ratio:.1f}:1)")
        strategies.extend([
            "1. 🎯 WEIGHTED LOSS FUNCTION",
            "2. 📈 LIGHT DATA AUGMENTATION",
            "3. ⚖️  BALANCED SAMPLING"
        ])
    else:
        print(f"✅ ACCEPTABLE IMBALANCE ({imbalance_ratio:.1f}:1)")
        strategies.extend([
            "1. 🎯 WEIGHTED LOSS FUNCTION (Optional)",
            "2. 📊 MONITOR per-class metrics"
        ])
    
    for strategy in strategies:
        print(f"   {strategy}")
    
    return strategies

def plot_class_distribution(class_counts, save_path=None):
    """Plot class distribution"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Bar plot
    bars = ax1.bar(classes, counts, color=['red', 'green', 'blue', 'orange', 'purple'])
    ax1.set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Samples')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    total = sum(counts)
    percentages = [count/total*100 for count in counts]
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    wedges, texts, autotexts = ax2.pie(counts, labels=classes, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.suptitle('ECG Class Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Class distribution plot saved to {save_path}")
    
    plt.show()

def create_balanced_sampler(labels, strategy='undersample'):
    """Create balanced sampler for DataLoader"""
    
    from torch.utils.data import WeightedRandomSampler
    
    if strategy == 'undersample':
        # Find the minimum class count
        class_counts = Counter(labels)
        min_count = min(class_counts.values())
        
        # Create balanced indices
        balanced_indices = []
        for class_label in class_counts.keys():
            class_indices = [i for i, label in enumerate(labels) if label == class_label]
            # Sample min_count samples from each class
            balanced_indices.extend(np.random.choice(class_indices, min_count, replace=False))
        
        print(f"\n⬇️  Undersampling Strategy:")
        print(f"   Target samples per class: {min_count:,}")
        print(f"   Total balanced samples: {len(balanced_indices):,}")
        
        return balanced_indices
    
    elif strategy == 'weighted':
        # Create sample weights
        class_counts = Counter(labels)
        sample_weights = [1.0 / class_counts[label] for label in labels]
        
        print(f"\n⚖️  Weighted Sampling Strategy:")
        print(f"   Sample weights range: {min(sample_weights):.4f} - {max(sample_weights):.4f}")
        
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def main():
    """Main function to demonstrate imbalance handling"""
    
    print("🔄 ECG Class Imbalance Handler")
    print("="*50)
    
    # Example with your actual distribution
    # You would load your actual labels here
    print("📋 This script demonstrates how to handle class imbalance.")
    print("   Replace the example data with your actual labels.")
    
    # Example distribution (replace with your actual data)
    example_labels = ['N'] * 122474 + ['Q'] * 16026 + ['V'] * 10540 + ['S'] * 4658 + ['F'] * 842
    
    # Analyze distribution
    class_counts, imbalance_ratio = analyze_class_distribution(example_labels)
    
    # Plot distribution
    plot_class_distribution(class_counts, 'outputs/reports/class_distribution.png')
    
    # Compute class weights
    class_weights = compute_class_weights(example_labels, method='balanced')
    
    # Suggest strategies
    strategies = suggest_balancing_strategies(imbalance_ratio, class_counts)
    
    print(f"\n✅ Analysis completed!")
    print(f"📁 Check outputs/reports/ for visualization")

if __name__ == "__main__":
    main()
