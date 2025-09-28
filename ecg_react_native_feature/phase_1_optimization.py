import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import time

print('🚀 PHASE 1: ADVANCED MINORITY CLASS OPTIMIZATION')
print('=' * 55)

# Phase 1 Implementation - Focus on V, S, F class improvement
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def advanced_ecg_augmentation(X, y, target_class, augmentation_factor=3):
    """Advanced ECG-specific augmentation for minority classes"""
    print(f'🔄 Generating {augmentation_factor}x augmented samples for class {target_class}')
    
    # Find samples of target class
    class_mask = (y == target_class)
    class_samples = X[class_mask]
    
    if len(class_samples) == 0:
        return X, y
    
    augmented_samples = []
    augmented_labels = []
    
    for _ in range(augmentation_factor):
        for sample in class_samples:
            # 1. Time warping (slight stretching/compression)
            if np.random.random() > 0.5:
                warp_factor = np.random.uniform(0.9, 1.1)
                indices = np.arange(len(sample))
                warped_indices = np.interp(indices, indices * warp_factor, indices)
                warped_sample = np.interp(warped_indices, indices, sample)
            else:
                warped_sample = sample.copy()
            
            # 2. Amplitude scaling
            if np.random.random() > 0.5:
                scale_factor = np.random.uniform(0.8, 1.2)
                warped_sample *= scale_factor
            
            # 3. Gaussian noise injection
            if np.random.random() > 0.5:
                noise_std = np.std(warped_sample) * 0.05
                noise = np.random.normal(0, noise_std, len(warped_sample))
                warped_sample += noise
            
            # 4. Baseline drift simulation
            if np.random.random() > 0.5:
                drift = np.linspace(0, np.random.uniform(-0.1, 0.1), len(warped_sample))
                warped_sample += drift
            
            augmented_samples.append(warped_sample)
            augmented_labels.append(target_class)
    
    # Combine original and augmented data
    X_augmented = np.vstack([X, np.array(augmented_samples)])
    y_augmented = np.concatenate([y, np.array(augmented_labels)])
    
    print(f'✅ Added {len(augmented_samples)} augmented samples for class {target_class}')
    return X_augmented, y_augmented

def calculate_class_weights(y, target_classes=['V', 'S', 'F']):
    """Calculate class weights with focus on minority classes"""
    unique_classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    # Enhanced weights for target minority classes
    class_weights = {}
    for cls, count in zip(unique_classes, counts):
        base_weight = total_samples / (len(unique_classes) * count)
        if cls in target_classes:
            # Give extra weight to difficult minority classes
            enhanced_weight = base_weight * 2.0
        else:
            enhanced_weight = base_weight
        class_weights[cls] = enhanced_weight
    
    print(f'📊 Class weights: {class_weights}')
    return class_weights

def create_minority_class_dataset():
    """Create optimized dataset focusing on minority classes"""
    print('📊 Loading and preparing data...')
    
    # Load data (using your existing paths)
    try:
        X = np.load('../mod/combined_ecg_final/test/segments.npy')
        y = np.load('../mod/combined_ecg_final/test/labels.npy')
        print(f'✅ Loaded {len(X)} samples')
    except:
        print('❌ Could not load data')
        return None, None
    
    # Show original distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f'\n📈 Original distribution:')
    for label, count in zip(unique_labels, counts):
        print(f'  {label}: {count:,} ({count/len(y)*100:.1f}%)')
    
    # Apply advanced augmentation to minority classes
    minority_classes = ['V', 'S', 'F']  # Focus on problematic classes
    
    X_enhanced = X.copy()
    y_enhanced = y.copy()
    
    for target_class in minority_classes:
        # Convert string labels to check
        class_indices = np.where(y == target_class)[0]
        if len(class_indices) > 0:
            # Determine augmentation factor based on class size
            class_count = len(class_indices)
            if class_count < 500:
                aug_factor = 5  # Heavy augmentation for very rare classes
            elif class_count < 1000:
                aug_factor = 3  # Medium augmentation
            else:
                aug_factor = 2  # Light augmentation
            
            X_enhanced, y_enhanced = advanced_ecg_augmentation(
                X_enhanced, y_enhanced, target_class, aug_factor
            )
    
    # Show enhanced distribution
    unique_labels_new, counts_new = np.unique(y_enhanced, return_counts=True)
    print(f'\n📈 Enhanced distribution:')
    for label, count in zip(unique_labels_new, counts_new):
        print(f'  {label}: {count:,} ({count/len(y_enhanced)*100:.1f}%)')
    
    return X_enhanced, y_enhanced

def quick_minority_class_test():
    """Quick test to validate minority class improvements"""
    print('\n🧪 QUICK MINORITY CLASS VALIDATION')
    print('=' * 40)
    
    # Load your working model architecture
    from simple_validation import MobileNetV1_ECG_Robust, DepthwiseSeparableConv
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV1_ECG_Robust(num_classes=5)
    
    # Load the robust model
    checkpoint = torch.load('../best_model_robust.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f'✅ Model loaded on {device}')
    
    # Create enhanced dataset
    X_enhanced, y_enhanced = create_minority_class_dataset()
    
    if X_enhanced is None:
        return
    
    # Quick validation on enhanced dataset
    mapping = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
    y_indices = np.array([mapping[label] for label in y_enhanced])
    
    # Test on a subset for speed
    sample_size = min(2000, len(X_enhanced))
    indices = np.random.choice(len(X_enhanced), sample_size, replace=False)
    
    X_test = X_enhanced[indices]
    y_test = y_indices[indices]
    
    # Run inference
    print(f'🔄 Testing on {sample_size} enhanced samples...')
    
    batch_size = 500
    all_predictions = []
    
    for i in range(0, len(X_test), batch_size):
        batch_X = X_test[i:i+batch_size]
        batch_tensor = torch.FloatTensor(batch_X).unsqueeze(1).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    accuracy = accuracy_score(y_test, all_predictions)
    
    print(f'📊 Enhanced Dataset Results:')
    print(f'  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    # Per-class performance on enhanced dataset
    class_names = ['N', 'V', 'S', 'F', 'Q']
    print(f'\nPer-Class Performance (Enhanced):')
    for i, class_name in enumerate(class_names):
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(all_predictions[class_mask] == i)
            class_count = np.sum(class_mask)
            print(f'  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.1f}%) - {class_count} samples')

# Implementation recommendations
def phase_1_implementation_plan():
    """Detailed Phase 1 implementation plan"""
    print('\n🎯 PHASE 1 IMPLEMENTATION PLAN')
    print('=' * 35)
    
    steps = [
        {
            'step': 1,
            'title': 'Advanced Data Augmentation',
            'description': 'Implement ECG-specific augmentation for V, S, F classes',
            'code_file': 'advanced_augmentation.py',
            'expected_improvement': '+3-5% overall accuracy'
        },
        {
            'step': 2, 
            'title': 'Focal Loss Implementation',
            'description': 'Replace CrossEntropy with Focal Loss (gamma=2.0)',
            'code_file': 'focal_loss_training.py',
            'expected_improvement': '+2-4% minority class recall'
        },
        {
            'step': 3,
            'title': 'Class-Weighted Training',
            'description': 'Enhanced class weights for V, S, F classes',
            'code_file': 'weighted_training.py', 
            'expected_improvement': '+1-3% minority class precision'
        },
        {
            'step': 4,
            'title': 'Progressive Training',
            'description': 'Train on Normal first, then fine-tune on minorities',
            'code_file': 'progressive_training.py',
            'expected_improvement': '+2-3% overall stability'
        }
    ]
    
    for step_info in steps:
        print(f"\nStep {step_info['step']}: {step_info['title']}")
        print(f"  Description: {step_info['description']}")
        print(f"  Implementation: {step_info['code_file']}")
        print(f"  Expected: {step_info['expected_improvement']}")
    
    print(f"\n🎯 PHASE 1 TARGET: 83-85% Overall Accuracy")
    print(f"   Current: 79.83% → Target: 83-85%")
    print(f"   Focus: V (30% → 50%), S (25% → 45%), F (25% → 40%)")

if __name__ == "__main__":
    # Run Phase 1 optimization
    phase_1_implementation_plan()
    
    print(f"\n🚀 Ready to start Phase 1 optimization!")
    print(f"Run quick_minority_class_test() to validate enhanced dataset")
    
    # Optionally run the quick test
    response = input(f"\nRun quick minority class validation? (y/n): ")
    if response.lower() == 'y':
        quick_minority_class_test()
    
    print(f"\n✅ Phase 1 setup complete!")
    print(f"Next: Implement focal loss training for minority classes")
