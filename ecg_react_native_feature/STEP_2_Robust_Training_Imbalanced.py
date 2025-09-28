itvate """
STEP 2: Robust Training for SEVERELY IMBALANCED ECG Data
Handles 65.6:1 class imbalance with advanced techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import os
from tqdm import tqdm
import time

# Import from parent directory
import sys
import os

# Get the current directory and add the mod directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
mod_dir = os.path.join(current_dir, '..', 'mod')
if mod_dir not in sys.path:
    sys.path.insert(0, mod_dir)

try:
    from models import MobileNetV1_1D
except ImportError:
    # Alternative path if the above doesn't work
    models_path = os.path.join(current_dir, 'mod')
    if models_path not in sys.path:
        sys.path.insert(0, models_path)
    from models import MobileNetV1_1D

class ECGDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            # Add noise for data augmentation
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            
            # Random scaling
            scale = torch.rand(1) * 0.2 + 0.9  # 0.9 to 1.1
            x = x * scale
            
        return x, y

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
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

class ImbalancedTrainer:
    def __init__(self, num_classes=5, device=None):
        # GPU detection and optimization
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"   PyTorch version: {torch.__version__}")
                print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                
                # Optimize GPU settings for laptop GPU
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Set memory allocation strategy for RTX 3050
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                
                # Test GPU with small tensor
                test_tensor = torch.randn(100, 100).to(device)
                test_result = torch.mm(test_tensor, test_tensor)
                print(f"   GPU test: {test_result.shape} ✅")
                
            else:
                device = 'cpu'
                print("⚠️  No GPU detected, using CPU")
                print("   To enable GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        
        self.device = device
        self.num_classes = num_classes
        self.class_names = ['N', 'V', 'S', 'F', 'Q']
        
        # Model with stronger regularization
        self.model = MobileNetV1_1D(
            num_classes=num_classes,
            width_multiplier=0.6,
            dropout_rate=0.5  # Increased dropout
        ).to(device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = total_params * 4 / 1024 / 1024
        print(f"📱 Model created with {total_params:,} parameters ({model_size_mb:.2f} MB)")
        print(f"🖥️  Using device: {device}")
    
    def calculate_class_weights(self, y):
        """Calculate class weights for imbalanced data"""
        class_counts = np.bincount(y)
        total_samples = len(y)
        
        # Balanced class weights
        weights = total_samples / (self.num_classes * class_counts)
        
        # Smooth weights to prevent extreme values
        weights = np.clip(weights, 0.1, 10.0)
        
        print(f"\n⚖️  Class weights calculated:")
        for i, (name, weight) in enumerate(zip(self.class_names, weights)):
            print(f"   {name}: {weight:.2f}")
            
        return torch.FloatTensor(weights).to(self.device)
    
    def create_balanced_sampler(self, y):
        """Create weighted sampler for balanced training"""
        class_counts = np.bincount(y)
        weights = 1.0 / class_counts
        sample_weights = weights[y]
        
        # Convert to list for PyTorch compatibility
        sample_weights_list = sample_weights.tolist()
        
        sampler = WeightedRandomSampler(
            weights=sample_weights_list,
            num_samples=len(sample_weights_list),
            replacement=True
        )
        
        return sampler
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training', 
                   leave=False, dynamic_ncols=True)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total
            current_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        pbar.close()
        acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, acc
    
    def validate(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        acc = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        # Calculate F1 score (better metric for imbalanced data)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return avg_loss, acc, f1, all_preds, all_targets
    
    def train_with_cross_validation(self, X, y, k_folds=3):
        """Train with stratified cross-validation"""
        
        print(f"🔄 Starting {k_folds}-fold cross-validation training")
        print(f"   Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []
        
        # Progress bar for folds
        fold_iterator = enumerate(skf.split(X, y))
        fold_pbar = tqdm(fold_iterator, total=k_folds, desc="Cross-Validation Folds", 
                        dynamic_ncols=True)
        
        for fold, (train_idx, val_idx) in fold_pbar:
            print(f"\n{'='*60}")
            print(f"🔥 FOLD {fold + 1}/{k_folds}")
            print(f"{'='*60}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"   Train: {len(X_train):,} samples")
            print(f"   Val:   {len(X_val):,} samples")
            
            # Check class distribution in training set
            train_counts = np.bincount(y_train, minlength=self.num_classes)
            print(f"   Train distribution: {train_counts}")
            
            # Create datasets
            train_dataset = ECGDataset(X_train, y_train, augment=True)
            val_dataset = ECGDataset(X_val, y_val, augment=False)
            
            # Create balanced sampler
            sampler = self.create_balanced_sampler(y_train)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=32, 
                sampler=sampler,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=64, 
                shuffle=False,
                num_workers=0
            )
            
            # Initialize model for this fold
            model = MobileNetV1_1D(
                num_classes=self.num_classes,
                width_multiplier=0.6,
                dropout_rate=0.5
            ).to(self.device)
            
            # Calculate class weights and create loss function
            class_weights = self.calculate_class_weights(y_train)
            
            # Use Focal Loss for severe imbalance
            criterion = FocalLoss(alpha=class_weights, gamma=2.0)
            
            # Optimizer with weight decay
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=0.001, 
                weight_decay=0.01
            )
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
            
            # Training loop with progress bar
            best_f1 = 0
            patience_counter = 0
            patience = 10
            max_epochs = 50
            
            train_losses = []
            val_losses = []
            val_f1_scores = []
            
            # Progress bar for epochs
            epoch_pbar = tqdm(range(max_epochs), desc=f"Fold {fold+1} Training", 
                             leave=False, dynamic_ncols=True)
            
            for epoch in epoch_pbar:
                # Train
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, optimizer, criterion, epoch
                )
                
                # Validate
                val_loss, val_acc, val_f1, _, _ = self.validate(
                    model, val_loader, criterion
                )
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_f1_scores.append(val_f1)
                
                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'Train_Loss': f'{train_loss:.3f}',
                    'Val_F1': f'{val_f1:.3f}',
                    'Best_F1': f'{best_f1:.3f}'
                })
                
                # Step scheduler
                scheduler.step(val_loss)
                
                # Early stopping based on F1 score
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                    # Save best model for this fold
                    torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        epoch_pbar.set_description(f"Fold {fold+1} Early Stop")
                        break
            
            epoch_pbar.close()
            
            # Load best model and final evaluation
            model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
            
            final_loss, final_acc, final_f1, preds, targets = self.validate(
                model, val_loader, criterion
            )
            
            fold_results.append({
                'fold': fold + 1,
                'accuracy': final_acc,
                'f1_score': final_f1,
                'loss': final_loss
            })
            
            print(f"\n  🎯 Fold {fold + 1} Results:")
            print(f"     Accuracy: {final_acc:.2f}%")
            print(f"     F1 Score: {final_f1:.4f}")
            print(f"     Loss: {final_loss:.4f}")
            
            # Classification report for this fold
            fold_pbar.set_postfix({
                'Fold': f'{fold+1}/{k_folds}',
                'F1': f'{final_f1:.4f}',
                'Acc': f'{final_acc:.2f}%'
            })
        
        fold_pbar.close()
        
        # Cross-validation summary
        print(f"\n{'='*60}")
        print(f"📊 CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        accuracies = [r['accuracy'] for r in fold_results]
        f1_scores = [r['f1_score'] for r in fold_results]
        
        print(f"Average Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        
        # Choose best fold
        best_fold_idx = np.argmax(f1_scores)
        best_fold = fold_results[best_fold_idx]
        
        print(f"\n🏆 Best Fold: {best_fold['fold']} (F1: {best_fold['f1_score']:.4f})")
        
        # Copy best model
        import shutil
        shutil.copy(f'best_model_fold_{best_fold_idx}.pth', 'best_model_robust.pth')
        
        # Clean up fold models
        for i in range(k_folds):
            if os.path.exists(f'best_model_fold_{i}.pth'):
                os.remove(f'best_model_fold_{i}.pth')
        
        print(f"✅ Best model saved as: best_model_robust.pth")
        
        return fold_results

def main():
    start_time = time.time()
    
    print("🚀 ROBUST ECG TRAINING FOR SEVERE IMBALANCE")
    print("="*60)
    
    # Load original data (not SMOTE)
    print("📁 Loading original ECG data...")
    X = np.load('../mod/combined_ecg_final/X_final_combined.npy')
    y = np.load('../mod/combined_ecg_final/y_final_combined.npy')
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Original class distribution:")
    class_names = ['N', 'V', 'S', 'F', 'Q']
    label_to_idx = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
    
    for i, (cls, count) in enumerate(zip(unique, counts)):
        if i < len(class_names):
            name = class_names[i]
        else:
            name = f'Class_{cls}'
        percentage = (count/len(y)) * 100
        print(f"   {name}: {count:,} ({percentage:.1f}%)")
    
    # Convert string labels to integers
    print(f"\n🔧 Converting string labels to integers...")
    y_int = np.array([label_to_idx[label] for label in y])
    print(f"   Label mapping: {label_to_idx}")
    print(f"   Integer labels shape: {y_int.shape}")
    
    # Normalize data to [-1, 1]
    print(f"\n🔧 Normalizing data to [-1, 1] range...")
    X_min, X_max = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # Avoid division by zero
    X = 2 * (X - X_min) / X_range - 1
    
    print(f"   Data range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Initialize trainer
    trainer = ImbalancedTrainer()
    
    # Train with cross-validation
    print(f"\n⏱️  Starting training...")
    training_start = time.time()
    
    results = trainer.train_with_cross_validation(X, y_int, k_folds=3)
    
    training_time = time.time() - training_start
    total_time = time.time() - start_time
    
    # Save results
    with open('imbalanced_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Training time: {training_time/60:.1f} minutes")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📊 Results saved to: imbalanced_training_results.json")
    print(f"🤖 Best model saved to: best_model_robust.pth")

if __name__ == "__main__":
    main()
