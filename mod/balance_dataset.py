#!/usr/bin/env python3
"""
SMOTE-Only Balancing for ECG Dataset - No Undersampling
"""

import numpy as np
from pathlib import Path
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt

def apply_smote_only_balancing():
    print("🎯 APPLYING SMOTE-ONLY BALANCING (No Undersampling)")
    print("="*60)
    
    # Load the combined dataset
    print("1. Loading combined dataset...")
    X = np.load('combined_ecg_final/X_final_combined.npy')
    y = np.load('combined_ecg_final/y_final_combined.npy')
    
    print(f"   Original shape: {X.shape}, {y.shape}")
    
    # Check current class distribution
    class_counts = Counter(y)
    print("\n2. Current class distribution:")
    for cls in ['F', 'N', 'Q', 'S', 'V']:
        count = class_counts.get(cls, 0)
        pct = (count / len(y)) * 100
        print(f"   {cls}: {count:,} samples ({pct:.1f}%)")
    
    # Calculate imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"   🔴 Current imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # Define target counts - ONLY OVERSAMPLE MINORITY CLASSES
    print("\n3. Setting target counts for balancing...")
    target_counts = {
        'N': 167289,  # Keep N class as is (NO undersampling)
        'F': 15000,   # Increase F-class (oversampling)
        'S': 12000,   # Increase S-class (oversampling)
        'V': 15000,   # Increase V-class (oversampling)
        'Q': 21859    # Keep Q-class same
    }
    
    print("   Target distribution:")
    for cls in ['F', 'N', 'Q', 'S', 'V']:
        current = class_counts.get(cls, 0)
        target = target_counts[cls]
        action = "↑" if target > current else "→"
        print(f"   {cls}: {current:,} {action} {target:,} samples")
    
    # Apply SMOTE only to classes that need oversampling
    print("\n4. Applying SMOTE to minority classes...")
    
    # Separate classes that need SMOTE
    classes_to_oversample = []
    sampling_strategy = {}
    
    for cls in ['F', 'S', 'V']:
        current_count = class_counts.get(cls, 0)
        target_count = target_counts[cls]
        
        if target_count > current_count:
            classes_to_oversample.append(cls)
            sampling_strategy[cls] = target_count
            print(f"   Will oversample {cls}: {current_count:,} → {target_count:,}")
    
    if not classes_to_oversample:
        print("   No classes need oversampling!")
        return X, y
    
    # Get indices for classes that need SMOTE
    smote_indices = []
    for cls in classes_to_oversample:
        indices = np.where(y == cls)[0]
        smote_indices.extend(indices)
    
    # Get indices for classes that don't need SMOTE
    other_classes = [cls for cls in ['N', 'Q'] if cls not in classes_to_oversample]
    other_indices = []
    for cls in other_classes:
        indices = np.where(y == cls)[0]
        other_indices.extend(indices)
    
    # Apply SMOTE only to the classes that need it
    X_smote = X[smote_indices]
    y_smote = y[smote_indices]
    
    smote = SMOTE(sampling_strategy=sampling_strategy, 
                 random_state=42, 
                 k_neighbors=3)
    
    X_resampled, y_resampled = smote.fit_resample(X_smote, y_smote)
    
    print(f"   After SMOTE: {X_resampled.shape}, {y_resampled.shape}")
    
    # Combine with classes that didn't need SMOTE
    X_other = X[other_indices]
    y_other = y[other_indices]
    
    X_balanced = np.vstack([X_other, X_resampled])
    y_balanced = np.concatenate([y_other, y_resampled])
    
    # Shuffle the dataset
    shuffle_indices = np.random.permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]
    
    print(f"\n5. Final balanced dataset: {X_balanced.shape}, {y_balanced.shape}")
    
    # Check new distribution
    balanced_counts = Counter(y_balanced)
    print("\n6. New class distribution:")
    for cls in ['F', 'N', 'Q', 'S', 'V']:
        count = balanced_counts.get(cls, 0)
        pct = (count / len(y_balanced)) * 100
        print(f"   {cls}: {count:,} samples ({pct:.1f}%)")
    
    new_imbalance = max(balanced_counts.values()) / min(balanced_counts.values())
    print(f"   ✅ New imbalance ratio: {new_imbalance:.1f}:1")
    print(f"   🎯 Improvement: {imbalance_ratio:.1f}:1 → {new_imbalance:.1f}:1")
    
    return X_balanced, y_balanced

def calculate_class_weights(y):
    """Calculate class weights for weighted loss function"""
    print("\n7. Calculating class weights for training...")
    
    class_counts = Counter(y)
    total_samples = len(y)
    
    # Calculate weights (inverse frequency)
    class_weights = {}
    for cls, count in class_counts.items():
        weight = total_samples / (len(class_counts) * count)
        class_weights[cls] = weight
    
    print("   Class weights (for weighted loss):")
    for cls, weight in class_weights.items():
        print(f"   {cls}: {weight:.3f}")
    
    return class_weights

def create_balanced_splits(X, y, output_dir="./balanced_ecg_smote"):
    """Create train/val/test splits from balanced data"""
    print("\n8. Creating balanced train/val/test splits...")
    
    # Stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save balanced dataset
    np.save(output_path / "X_balanced.npy", X)
    np.save(output_path / "y_balanced.npy", y)
    
    # Save splits
    splits_dir = output_path / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    np.save(splits_dir / "X_train.npy", X_train)
    np.save(splits_dir / "y_train.npy", y_train)
    np.save(splits_dir / "X_val.npy", X_val)
    np.save(splits_dir / "y_val.npy", y_val)
    np.save(splits_dir / "X_test.npy", X_test)
    np.save(splits_dir / "y_test.npy", y_test)
    
    # Print split summary
    split_info = {
        'Train': (X_train, y_train),
        'Validation': (X_val, y_val),
        'Test': (X_test, y_test)
    }
    
    print("   📊 Split distribution:")
    for split_name, (X_split, y_split) in split_info.items():
        counts = Counter(y_split)
        print(f"   {split_name}: {len(y_split):,} samples")
        for cls in sorted(counts.keys()):
            print(f"      {cls}: {counts[cls]:,}")
    
    return output_path

def plot_comparison(original_y, balanced_y):
    """Plot before/after comparison"""
    plt.figure(figsize=(12, 5))
    
    # Before balancing
    plt.subplot(1, 2, 1)
    orig_counts = Counter(original_y)
    plt.bar(orig_counts.keys(), orig_counts.values(), color='red', alpha=0.7)
    plt.title('Before Balancing\n(Imbalance: 65.6:1)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # After balancing  
    plt.subplot(1, 2, 2)
    balanced_counts = Counter(balanced_y)
    plt.bar(balanced_counts.keys(), balanced_counts.values(), color='green', alpha=0.7)
    plt.title('After SMOTE Balancing\n(No Undersampling)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('balancing_comparison_smote.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main balancing pipeline"""
    print("🚀 ECG DATASET BALANCING (SMOTE-ONLY)")
    print("="*60)
    
    # Apply SMOTE-only balancing
    X_balanced, y_balanced = apply_smote_only_balancing()
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_balanced)
    
    # Create balanced splits
    output_path = create_balanced_splits(X_balanced, y_balanced)
    
    # Save class weights
    weights_file = output_path / "class_weights.json"
    with open(weights_file, 'w') as f:
        json.dump(class_weights, f, indent=2)
    
    # Create comparison plot
    original_y = np.load('combined_ecg_final/y_final_combined.npy')
    plot_comparison(original_y, y_balanced)
    
    print(f"\n🎉 SMOTE-ONLY BALANCING COMPLETE!")
    print(f"📁 Output saved to: {output_path}")
    print(f"⚖️  Class weights saved to: {weights_file}")
    print(f"📈 Comparison plot saved: balancing_comparison_smote.png")
    
    print(f"\n🤖 NEXT STEPS:")
    print(f"1. Use X_balanced.npy and y_balanced.npy for training")
    print(f"2. Apply class weights in your loss function")
    print(f"3. Train your model on the balanced dataset!")

if __name__ == "__main__":
    main()