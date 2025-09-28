#!/usr/bin/env python3
"""
Optimized MobileNetV1 1D Trainer for Real-Time ECG Classification
Handles class imbalance, mixed precision training, and real-time optimization
FIXED: Overfitting issues and JSON serialization bug
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import ndimage  # For signal smoothing
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')
from models import create_model

class OptimizedMobileNetTrainer:
    """
    Optimized trainer for MobileNetV1 1D with real-time focus and overfitting prevention
    Features:
    - Mixed precision training (2x faster)
    - Advanced class balancing
    - Real-time inference optimization
    - Enhanced regularization to prevent overfitting
    - Model quantization ready
    - Memory efficient training
    """
    
    def __init__(self, model_name='mobilenet', config=None):
        self.model_name = model_name
        self.config = config or self.get_mobilenet_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mixed precision training
        self.use_amp = torch.cuda.is_available() and self.config.get('use_mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.class_weights = None
        self.label_encoder = None
        
        # Training metrics
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_balanced_acc': [],
            'epochs': [], 'lr': [], 'inference_time': []
        }
        
        # Best model tracking
        self.best_metrics = {
            'val_acc': 0.0, 'val_f1': 0.0, 'val_balanced_acc': 0.0,
            'epoch': 0, 'inference_time': float('inf')
        }
        
        print(f"🚀 Optimized MobileNet Trainer (Anti-Overfitting)")
        print(f"📱 Device: {self.device}")
        print(f"⚡ Mixed Precision: {self.use_amp}")
        print(f"🧠 Model: {model_name}")
    
    def get_mobilenet_config(self):
        """Approach 2: Generalization-First Configuration (Tight Gaps)"""
        return {
        # Core training - ultra conservative for tight generalization
        'batch_size': 24,  # Smaller batch for better generalization
        'learning_rate': 0.0005,  # Much lower for stability
        'epochs': 120,  # Reduce to prevent overtraining
        'weight_decay': 8e-4,  # Higher regularization
        
        # Scheduler - very conservative
        'scheduler_type': 'cosine_warm_restart',
        'patience': 6,  # Earlier LR reduction
        'factor': 0.3,  # More aggressive reduction
        'min_lr': 1e-9,  # Lower minimum
        'warmup_epochs': 3,  # Shorter warmup
        
        # Ultra regularization for tight gaps
        'dropout_rate': 0.5,  # High dropout
        'use_layer_dropout': True,
        'layer_dropout_rate': 0.2,  # NEW: Additional layer dropout
        'label_smoothing': 0.15,  # Higher smoothing
        'gradient_clip_norm': 0.5,  # Tighter clipping
        
        # Class imbalance handling
        'use_weighted_loss': True,
        'use_focal_loss': True,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'use_balanced_sampling': True,
        'sampling_strategy': 'weighted',
        
        # Very aggressive early stopping for generalization
        'early_stopping_patience': 12,  # Reduced patience
        'early_stopping_metric': 'generalization_gap',  # NEW: Gap-based stopping
        'max_generalization_gap': 4.0,  # NEW: Hard gap limit
        'save_best_only': True,
        
        # Model size for generalization
        'width_multiplier': 0.55,  # Even smaller
        
        # Real-time optimization
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 1,
        
        # Enhanced augmentation for regularization
        'use_augmentation': True,
        'augmentation_prob': 0.7,  # Higher probability
        'noise_factor': 0.03,  # More noise
        'time_shift_max': 20,  # Larger shifts
        'amplitude_scale_range': (0.7, 1.3),  # Wider range
        'time_warp_sigma': 0.3,
        'use_mixup': True,  # NEW: Advanced augmentation
        'mixup_alpha': 0.2,
        'use_cutmix': True,  # NEW: Advanced augmentation
        'cutmix_alpha': 0.3,
        
        # Validation strategy
        'validation_split': 0.2,
        'stratified_split': True,
        
        # NEW: Gap monitoring targets
        'target_accuracy_gap': (2.0, 5.0),  # Target range
        'target_f1_gap': (1.0, 4.0),  # Target range
        'strict_gap_enforcement': True
    }
    
    def load_balanced_data(self, data_dir='balanced_ecg_smote/splits'):
        """Load and analyze balanced ECG data with enhanced validation"""
        print(f"📊 Loading balanced ECG data from {data_dir}...")
        
        # Load all splits
        splits = ['train', 'val', 'test']
        data = {}
        
        for split in splits:
            segments = np.load(f'{data_dir}/X_{split}.npy')
            labels = np.load(f'{data_dir}/y_{split}.npy')
            
            # Handle string labels
            if labels.dtype.kind in ['U', 'S']:
                if self.label_encoder is None:
                    self.label_encoder = LabelEncoder()
                    labels = self.label_encoder.fit_transform(labels)
                else:
                    labels = self.label_encoder.transform(labels)
            
            data[split] = (segments, labels)
            print(f"📈 {split.title()}: {len(segments):,} segments")
        
        # Store data
        self.train_segments, self.train_labels = data['train']
        self.val_segments, self.val_labels = data['val']
        self.test_segments, self.test_labels = data['test']
        
        # Set dimensions
        self.input_size = self.train_segments.shape[1]
        self.num_classes = len(np.unique(self.train_labels))
        
        print(f"\n📏 Input size: {self.input_size}")
        print(f"🏷️ Number of classes: {self.num_classes}")
        
        # Analyze class distribution
        self.analyze_class_distribution()
        
        # Print label mapping if available
        if self.label_encoder:
            print(f"\n🔤 Label mapping:")
            for i, class_name in enumerate(self.label_encoder.classes_):
                print(f"   {i} → {class_name}")
    
    def analyze_class_distribution(self):
        """Detailed analysis of class distribution across all splits"""
        print(f"\n📊 DATASET ANALYSIS")
        print("="*50)
        
        for split_name, labels in [('Train', self.train_labels), 
                                 ('Validation', self.val_labels), 
                                 ('Test', self.test_labels)]:
            
            class_counts = Counter(labels)
            total = len(labels)
            
            print(f"\n{split_name} Distribution ({total:,} samples):")
            
            for cls in sorted(class_counts.keys()):
                count = class_counts[cls]
                pct = (count / total) * 100
                class_name = self.label_encoder.inverse_transform([cls])[0] if self.label_encoder else cls
                
                # Visual bar
                bar_length = int(pct / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                print(f"   {class_name} (class {cls}): {count:>7,} ({pct:5.1f}%) |{bar}|")
            
            # Calculate imbalance metrics
            if len(class_counts) > 1:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                imbalance_ratio = max_count / min_count
                
                if imbalance_ratio > 100:
                    status = "🚨 SEVERE"
                elif imbalance_ratio > 10:
                    status = "⚠️  HIGH"
                else:
                    status = "✅ ACCEPTABLE"
                
                print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1 {status}")
    
    def setup_enhanced_loss(self):
        """Setup enhanced loss function with label smoothing"""
        print(f"🔧 Setting up enhanced loss function with regularization...")
        
        class EnhancedFocalLoss(nn.Module):
            def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.label_smoothing = label_smoothing
                self.num_classes = len(alpha) if alpha is not None else None
            
            def forward(self, inputs, targets):
                # Apply label smoothing
                if self.label_smoothing > 0 and self.num_classes:
                    # Create soft labels
                    soft_targets = torch.zeros_like(inputs)
                    soft_targets.fill_(self.label_smoothing / (self.num_classes - 1))
                    soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
                    
                    # Calculate loss with soft labels
                    log_probs = F.log_softmax(inputs, dim=1)
                    loss = -torch.sum(soft_targets * log_probs, dim=1)
                    
                    if self.alpha is not None:
                        alpha_weights = self.alpha.gather(0, targets)
                        loss = alpha_weights * loss
                    
                    # Apply focal weight
                    probs = F.softmax(inputs, dim=1)
                    target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                    focal_weight = (1 - target_probs) ** self.gamma
                    loss = focal_weight * loss
                    
                    return loss.mean()
                else:
                    # Standard focal loss
                    ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_loss = (1 - pt) ** self.gamma * ce_loss
                    return focal_loss.mean()
        
        # Setup class weights
        self.setup_class_weights()
        
        if self.class_weights is not None:
            weights_tensor = torch.tensor(
                [self.class_weights[i] for i in sorted(self.class_weights.keys())], 
                dtype=torch.float32, device=self.device
            )
            
            self.criterion = EnhancedFocalLoss(
                alpha=weights_tensor,
                gamma=self.config['focal_gamma'],
                label_smoothing=self.config.get('label_smoothing', 0.1)
            )
            print(f"🎯 Using Enhanced Focal Loss (gamma={self.config['focal_gamma']}, label_smoothing={self.config.get('label_smoothing', 0.1)})")
        else:
            self.criterion = EnhancedFocalLoss(
                gamma=self.config['focal_gamma'],
                label_smoothing=self.config.get('label_smoothing', 0.1)
            )
    
    def setup_class_weights(self):
        """Setup class weights from precomputed file or compute on-the-fly"""
        weights_path = Path("balanced_ecg_smote/class_weights.json")
        
        if weights_path.exists():
            print(f"📊 Loading precomputed class weights...")
            with open(weights_path, 'r') as f:
                loaded_weights = json.load(f)
            
            self.class_weights = {}
            for cls_str, weight in loaded_weights.items():
                if self.label_encoder and cls_str in self.label_encoder.classes_:
                    cls_int = list(self.label_encoder.classes_).index(cls_str)
                    self.class_weights[cls_int] = weight
                else:
                    try:
                        self.class_weights[int(cls_str)] = weight
                    except ValueError:
                        continue
        else:
            print(f"⚖️  Computing class weights from training data...")
            unique_classes = np.unique(self.train_labels)
            weights = compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=self.train_labels
            )
            self.class_weights = {cls: weight for cls, weight in zip(unique_classes, weights)}
        
        # Display weights
        print(f"📊 Class weights:")
        for cls in sorted(self.class_weights.keys()):
            class_name = self.label_encoder.inverse_transform([cls])[0] if self.label_encoder else cls
            print(f"   {class_name} (class {cls}): {self.class_weights[cls]:.3f}")
    
    def setup_model(self):
        """Setup optimized MobileNetV1 model with enhanced regularization"""
        print(f"🔧 Setting up MobileNetV1 1D model with anti-overfitting measures...")
        
        # Create model with enhanced regularization
        self.model = create_model(
            self.model_name, 
            input_size=self.input_size, 
            num_classes=self.num_classes,
            width_multiplier=self.config['width_multiplier'],
            dropout_rate=self.config['dropout_rate']
        )
        self.model.to(self.device)
        
        # Setup enhanced loss
        self.setup_enhanced_loss()
        
        # Setup optimizer with different learning rates and stronger regularization
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if 'classifier' in n], 
                'lr': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() if 'features' in n], 
                'lr': self.config['learning_rate'] * 0.1,
                'weight_decay': self.config['weight_decay'] * 2  # Higher regularization for features
            }
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            eps=1e-8,
            amsgrad=True
        )
        
        # Enhanced scheduler with more aggressive reduction
        if self.config['scheduler_type'] == 'cosine_warm_restart':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=15,  # Shorter restart period
                T_mult=2,   # Must be integer >= 1
                eta_min=self.config['min_lr']
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min' if self.config['early_stopping_metric'] == 'val_loss' else 'max',
                factor=self.config['factor'],
                patience=self.config['patience'],
                min_lr=self.config['min_lr'],
                verbose=True
            )
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        print(f"📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {model_size_mb:.2f} MB")
        print(f"   Target for mobile: < 5 MB {'✅' if model_size_mb < 5 else '⚠️'}")
        print(f"   Dropout rate: {self.config['dropout_rate']}")
        print(f"   Label smoothing: {self.config.get('label_smoothing', 0.0)}")
    
    def enhanced_augmentation(self, segments, labels):
        """Enhanced augmentation with Approach 2 settings"""
        if not self.config.get('use_augmentation', True):
            return segments, labels
        
        augmented_segments = []
        augmented_labels = []
        
        for segment, label in zip(segments, labels):
            # Always include original
            augmented_segments.append(segment)
            augmented_labels.append(label)
            
            # Apply augmentation with Approach 2 probability
            if np.random.random() < self.config.get('augmentation_prob', 0.7):
                aug_segment = segment.copy()
                
                # 1. Gaussian noise (higher for Approach 2)
                if np.random.random() < 0.5:
                    noise = np.random.normal(0, self.config['noise_factor'], segment.shape)
                    aug_segment += noise
                
                # 2. Time shifting (larger for Approach 2)
                if np.random.random() < 0.4:
                    shift = np.random.randint(-self.config['time_shift_max'], 
                                             self.config['time_shift_max'])
                    aug_segment = np.roll(aug_segment, shift)
                
                # 3. Amplitude scaling (wider range for Approach 2)
                if np.random.random() < 0.4:
                    scale_range = self.config.get('amplitude_scale_range', (0.7, 1.3))
                    scale = np.random.uniform(scale_range[0], scale_range[1])
                    aug_segment *= scale
                
                # 4. Time warping (stronger for Approach 2)
                if np.random.random() < 0.3:
                    sigma = self.config.get('time_warp_sigma', 0.3)
                    indices = np.arange(len(aug_segment))
                    warped_indices = indices + np.random.normal(0, sigma, len(indices))
                    warped_indices = np.clip(warped_indices, 0, len(indices)-1).astype(int)
                    aug_segment = aug_segment[warped_indices]
                
                # 5. Signal smoothing for generalization
                if np.random.random() < 0.2:
                    try:
                        from scipy import ndimage
                        aug_segment = ndimage.gaussian_filter1d(aug_segment, sigma=0.5)
                    except ImportError:
                        # Skip if scipy not available
                        pass
                
                # 6. Random masking
                if np.random.random() < 0.2:
                    mask_length = np.random.randint(5, 20)
                    mask_start = np.random.randint(0, len(aug_segment) - mask_length)
                    aug_segment[mask_start:mask_start + mask_length] = 0
                
                # Clip values to reasonable range
                aug_segment = np.clip(aug_segment, -10, 10)
                
                augmented_segments.append(aug_segment)
                augmented_labels.append(label)
        
        # Apply MixUp/CutMix if enabled
        if self.config.get('use_mixup', False) or self.config.get('use_cutmix', False):
            augmented_segments, augmented_labels = self.apply_mixup_cutmix(
                np.array(augmented_segments), np.array(augmented_labels)
            )
        
        return np.array(augmented_segments), np.array(augmented_labels)
    
    def apply_mixup_cutmix(self, segments, labels):
        """MixUp and CutMix for Approach 2"""
        if not (self.config.get('use_mixup', False) or self.config.get('use_cutmix', False)):
            return segments, labels
        
        mixed_segments = []
        mixed_labels = []
        
        for i in range(len(segments)):
            if np.random.random() < 0.3:  # 30% chance of mixing
                # Choose random partner
                j = np.random.randint(0, len(segments))
                
                if self.config.get('use_mixup', False) and np.random.random() < 0.5:
                    # MixUp
                    alpha = self.config.get('mixup_alpha', 0.2)
                    lam = np.random.beta(alpha, alpha)
                    mixed_segment = lam * segments[i] + (1 - lam) * segments[j]
                    mixed_segments.append(mixed_segment)
                    mixed_labels.append(labels[i])  # Keep original label
                    
                elif self.config.get('use_cutmix', False):
                    # CutMix (simplified for 1D)
                    alpha = self.config.get('cutmix_alpha', 0.3)
                    lam = np.random.beta(alpha, alpha)
                    
                    cut_length = int(lam * len(segments[i]))
                    cut_start = np.random.randint(0, len(segments[i]) - cut_length)
                    
                    mixed_segment = segments[i].copy()
                    mixed_segment[cut_start:cut_start + cut_length] = segments[j][cut_start:cut_start + cut_length]
                    
                    mixed_segments.append(mixed_segment)
                    mixed_labels.append(labels[i])  # Keep original label
            else:
                mixed_segments.append(segments[i])
                mixed_labels.append(labels[i])
        
        return np.array(mixed_segments), np.array(mixed_labels)
    
    
    def create_balanced_data_loaders(self):
        """Create optimized data loaders with enhanced augmentation and balancing"""
        print(f"📦 Creating balanced data loaders with enhanced augmentation...")
        
        # Apply enhanced augmentation only to minority classes
        minority_threshold = 0.15
        class_counts = Counter(self.train_labels)
        total_samples = len(self.train_labels)
        
        minority_classes = [cls for cls, count in class_counts.items() 
                           if count / total_samples < minority_threshold]
        
        if minority_classes and self.config.get('use_augmentation', True):
            print(f"🔄 Applying enhanced augmentation to minority classes: {minority_classes}")
            
            minority_mask = np.isin(self.train_labels, minority_classes)
            minority_segments = self.train_segments[minority_mask]
            minority_labels = self.train_labels[minority_mask]
            
            # Apply enhanced augmentation
            aug_segments, aug_labels = self.enhanced_augmentation(minority_segments, minority_labels)
            
            # Combine with original data
            self.train_segments = np.vstack([self.train_segments, aug_segments])
            self.train_labels = np.hstack([self.train_labels, aug_labels])
            
            print(f"   Added {len(aug_segments):,} augmented samples")
        
        # Create weighted sampler
        sampler = None
        if self.config.get('use_balanced_sampling', True):
            class_counts = Counter(self.train_labels)
            sample_weights = [1.0 / class_counts[label] for label in self.train_labels]
            sample_weights = np.array(sample_weights)
            sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
            
            sampler = WeightedRandomSampler(
                sample_weights, 
                len(sample_weights), 
                replacement=True
            )
            print(f"⚖️  Using weighted balanced sampling")
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(self.train_segments),
            torch.LongTensor(self.train_labels)
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(self.val_segments),
            torch.LongTensor(self.val_labels)
        )
        
        test_dataset = TensorDataset(
            torch.FloatTensor(self.test_segments),
            torch.LongTensor(self.test_labels)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"✅ Data loaders created:")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print(f"   Test batches: {len(self.test_loader)}")
    
    def train_epoch(self, epoch):
        """Enhanced training epoch with gradient clipping and regularization"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, 
                           desc=f"Epoch {epoch:3d}/{self.config['epochs']}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass with scaling and gradient clipping
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_norm', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.config['gradient_clip_norm'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.config['gradient_clip_norm'])
                
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Update progress
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted') * 100
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def validate_epoch(self, epoch):
        """Validation with inference timing"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                start_time = time.time()
                
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                inference_time = (time.time() - start_time) * 1000 / data.size(0)
                inference_times.append(inference_time)
                
                running_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted') * 100
        epoch_balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
        avg_inference_time = np.mean(inference_times)
        
        return epoch_loss, epoch_acc, epoch_f1, epoch_balanced_acc, avg_inference_time
    
    def train(self, save_dir='outputs/models'):
        """Main training loop with enhanced early stopping and gap monitoring for Approach 2"""
        print(f"\n🛡️ Starting Approach 2: Generalization-First Training...")
        print(f"🎯 Focus: Tight generalization gaps over peak performance")
        print(f"📊 Early stopping metric: {self.config['early_stopping_metric']}")
        print(f"🎯 Target gaps: Gen<{self.config.get('max_generalization_gap', 4.0)}%, Acc={self.config.get('target_accuracy_gap', (2.0, 5.0))}, F1={self.config.get('target_f1_gap', (1.0, 4.0))}")
        
        os.makedirs(save_dir, exist_ok=True)
        start_time = time.time()
        
        # Enhanced early stopping for Approach 2
        patience_counter = 0
        best_score = 0.0  # For F1 score
        best_generalization_gap = float('inf')
        
        # Load checkpoint for resuming training
        start_epoch = self.load_checkpoint_for_resume(f'{save_dir}/mobilenet_best.pth')
        
        # Warmup phase
        warmup_epochs = self.config.get('warmup_epochs', 3)
        
        for epoch in range(start_epoch, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            # Warmup learning rate schedule
            if epoch <= warmup_epochs:
                warmup_lr = self.config['learning_rate'] * (epoch / warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Training phase
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, val_f1, val_balanced_acc, inference_time = self.validate_epoch(epoch)
            
            # Get current learning rate for logging
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_balanced_acc'].append(val_balanced_acc)
            self.history['epochs'].append(epoch)
            self.history['lr'].append(current_lr)
            self.history['inference_time'].append(inference_time)
            
            # Calculate comprehensive gaps for Approach 2
            generalization_gap = train_acc - val_acc
            accuracy_gap = abs(train_acc - val_acc)
            f1_gap = abs(train_f1 - val_f1)
            
            # Approach 2: Strict gap requirements
            target_acc_gap = self.config.get('target_accuracy_gap', (2.0, 5.0))
            target_f1_gap = self.config.get('target_f1_gap', (1.0, 4.0))
            max_gen_gap = self.config.get('max_generalization_gap', 4.0)
            
            # Check if gaps are within target ranges
            gaps_within_target = (
                generalization_gap < max_gen_gap and
                target_acc_gap[0] <= accuracy_gap <= target_acc_gap[1] and
                target_f1_gap[0] <= f1_gap <= target_f1_gap[1]
            )
            
            # Approach 2: Model selection prioritizes generalization
            if self.config.get('strict_gap_enforcement', True):
                # Only save models that meet gap requirements
                is_best = gaps_within_target and val_f1 > best_score
                selection_criteria = "Gap-Constrained F1"
            else:
                # Alternative: Gap-penalized selection
                gap_penalty = max(0, (generalization_gap - 3) / 10)
                adjusted_score = val_f1 * (1 - 0.15 * gap_penalty)
                is_best = adjusted_score > best_score
                selection_criteria = "Gap-Penalized F1"
            
            # Update best model if criteria met
            if is_best:
                best_score = val_f1 if self.config.get('strict_gap_enforcement', True) else adjusted_score
                best_generalization_gap = generalization_gap
                
                self.best_metrics.update({
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_balanced_acc': val_balanced_acc,
                    'epoch': epoch,
                    'inference_time': inference_time,
                    'generalization_gap': generalization_gap,
                    'accuracy_gap': accuracy_gap,
                    'f1_gap': f1_gap,
                    'gaps_within_target': gaps_within_target
                })
                
                self.save_model(f'{save_dir}/mobilenet_best.pth')
                patience_counter = 0
                
                print(f"   🏆 NEW BEST MODEL SAVED!")
            else:
                patience_counter += 1
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Enhanced logging with gap status for Approach 2
            status = "BEST!" if is_best else f"({patience_counter}/{self.config['early_stopping_patience']})"
            real_time_status = "REAL-TIME" if inference_time < 50 else "SLOW"
            
            # Gap status indicators
            if gaps_within_target:
                gap_status = "✅ PERFECT GAPS"
            elif generalization_gap < max_gen_gap:
                gap_status = "⚠️ MINOR GAP ISSUE"
            else:
                gap_status = "🚨 GAP VIOLATION"
            
            print(f"\nEpoch {epoch:3d}/{self.config['epochs']} - {status} ({selection_criteria})")
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:5.2f}%, F1={train_f1:5.2f}%")
            print(f"   Valid: Loss={val_loss:.4f}, Acc={val_acc:5.2f}%, F1={val_f1:5.2f}%, BalAcc={val_balanced_acc:5.2f}%")
            print(f"   Gaps: Gen={generalization_gap:4.1f}%, Acc={accuracy_gap:4.1f}%, F1={f1_gap:4.1f}% - {gap_status}")
            print(f"   Target: Gen<{max_gen_gap}%, Acc={target_acc_gap[0]}-{target_acc_gap[1]}%, F1={target_f1_gap[0]}-{target_f1_gap[1]}%")
            print(f"   Inference: {inference_time:.2f}ms/sample {real_time_status}")
            print(f"   LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
            
            # Update scheduler after warmup
            if epoch > warmup_epochs:
                if self.config['scheduler_type'] == 'cosine_warm_restart':
                    self.scheduler.step()
                else:
                    self.scheduler.step(val_loss if self.config['early_stopping_metric'] == 'val_loss' else -val_f1)
            
            # Approach 2: Enhanced early stopping with strict gap monitoring
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"\n🛑 Early stopping: {patience_counter} epochs without gap-compliant improvement")
                print(f"   Best generalization gap achieved: {best_generalization_gap:.2f}%")
                break
                
            # Stop if gaps violate targets consistently
            if epoch > 20 and generalization_gap > max_gen_gap * 1.5:
                print(f"\n🚨 Stopping: Persistent gap violation!")
                print(f"   Current gap: {generalization_gap:.1f}% > threshold: {max_gen_gap * 1.5:.1f}%")
                print(f"   Recommendation: Further reduce model capacity or increase regularization")
                break
            
            # Stop if model shows severe overfitting signs
            if epoch > 30 and train_acc > val_acc + 20:
                print(f"\n🚨 Critical overfitting detected! Stopping training.")
                print(f"   Train accuracy ({train_acc:.1f}%) >> Val accuracy ({val_acc:.1f}%)")
                break
        
        # Training completion summary
        total_time = time.time() - start_time
        
        print(f"\n🏁 APPROACH 2 TRAINING COMPLETED!")
        print("="*60)
        print(f"⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"🏆 Best Results (Epoch {self.best_metrics['epoch']}):")
        print(f"   Validation Accuracy: {self.best_metrics['val_acc']:.2f}%")
        print(f"   Weighted F1-Score: {self.best_metrics['val_f1']:.2f}%")
        print(f"   Balanced Accuracy: {self.best_metrics['val_balanced_acc']:.2f}%")
        print(f"   Generalization Gap: {self.best_metrics.get('generalization_gap', 0):.2f}%")
        print(f"   Accuracy Gap: {self.best_metrics.get('accuracy_gap', 0):.2f}%")
        print(f"   F1 Gap: {self.best_metrics.get('f1_gap', 0):.2f}%")
        print(f"   Inference Time: {self.best_metrics['inference_time']:.2f}ms/sample")
        print(f"   Gaps Within Target: {'YES ✅' if self.best_metrics.get('gaps_within_target', False) else 'NO ❌'}")
        print(f"   Real-time Ready: {'YES ✅' if self.best_metrics['inference_time'] < 50 else 'NO ❌'}")
        
        # Approach 2 assessment
        final_gap = self.best_metrics.get('generalization_gap', 0)
        if final_gap < 2:
            print(f"\n🎯 OUTSTANDING: Excellent generalization achieved!")
        elif final_gap < 4:
            print(f"\n✅ SUCCESS: Target generalization gap achieved!")
        elif final_gap < 6:
            print(f"\n⚠️  ACCEPTABLE: Minor overfitting, but usable")
        else:
            print(f"\n❌ NEEDS IMPROVEMENT: Consider stronger regularization")
        
        # Save comprehensive results
        self.save_training_results(save_dir)
        
        return self.history
    
    def evaluate_test_set(self, model_path=None):
        """Comprehensive test set evaluation"""
        print(f"\nFINAL TEST SET EVALUATION")
        print("="*50)
        
        if model_path:
            print(f"Loading best model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        all_preds = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                
                start_time = time.time()
                
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                else:
                    output = self.model(data)
                
                inference_time = (time.time() - start_time) * 1000 / data.size(0)
                inference_times.append(inference_time)
                
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate comprehensive metrics
        test_acc = accuracy_score(all_targets, all_preds) * 100
        test_f1_weighted = f1_score(all_targets, all_preds, average='weighted') * 100
        test_f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
        test_balanced_acc = balanced_accuracy_score(all_targets, all_preds) * 100
        test_precision = precision_score(all_targets, all_preds, average='weighted') * 100
        test_recall = recall_score(all_targets, all_preds, average='weighted') * 100
        avg_inference_time = np.mean(inference_times)
        
        print(f"\nFINAL TEST RESULTS:")
        print(f"   Accuracy: {test_acc:.2f}%")
        print(f"   Weighted F1: {test_f1_weighted:.2f}%")
        print(f"   Macro F1: {test_f1_macro:.2f}%")
        print(f"   Balanced Accuracy: {test_balanced_acc:.2f}%")
        print(f"   Precision: {test_precision:.2f}%")
        print(f"   Recall: {test_recall:.2f}%")
        print(f"   Avg Inference: {avg_inference_time:.2f}ms/sample")
        print(f"   Real-time Ready: {'YES' if avg_inference_time < 50 else 'NO'}")
        
        # Detailed classification report
        if self.label_encoder:
            target_names = self.label_encoder.classes_
        else:
            target_names = [f"Class_{i}" for i in range(self.num_classes)]
        
        print(f"\nDETAILED CLASSIFICATION REPORT:")
        report = classification_report(all_targets, all_preds, 
                                     target_names=target_names, 
                                     digits=3)
        print(report)
        
        # Confusion matrix
        self.plot_confusion_matrix(all_targets, all_preds, target_names)
        
        return {
            'accuracy': test_acc,
            'f1_weighted': test_f1_weighted,
            'f1_macro': test_f1_macro,
            'balanced_accuracy': test_balanced_acc,
            'precision': test_precision,
            'recall': test_recall,
            'inference_time_ms': avg_inference_time,
            'real_time_ready': avg_inference_time < 50
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.model_name.upper()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        os.makedirs('outputs/plots', exist_ok=True)
        plt.savefig(f'outputs/plots/{self.model_name}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to outputs/plots/")
        plt.close()
    
    def optimize_for_deployment(self, model_path, output_dir='outputs/deployment'):
        """Optimize model for real-time deployment with fixed JSON serialization"""
        print(f"\nOPTIMIZING FOR DEPLOYMENT")
        print("="*50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load best model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 1. Export to TorchScript
        print(f"Converting to TorchScript...")
        dummy_input = torch.randn(1, self.input_size, device=self.device)
        
        try:
            traced_model = torch.jit.trace(self.model, dummy_input)
            script_path = f'{output_dir}/mobilenet_traced.pt'
            traced_model.save(script_path)
            
            with torch.no_grad():
                original_output = self.model(dummy_input)
                traced_output = traced_model(dummy_input)
                max_diff = torch.max(torch.abs(original_output - traced_output))
                print(f"   TorchScript conversion successful")
                print(f"   Max output difference: {max_diff:.2e}")
                
        except Exception as e:
            print(f"   TorchScript conversion failed: {e}")
        
        # 2. Model quantization
        print(f"Applying quantization...")
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model.cpu(), {nn.Linear}, dtype=torch.qint8
            )
            
            test_input = torch.randn(100, self.input_size)
            
            # Time original model
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(test_input)
            original_time = (time.time() - start_time) * 1000 / 100
            
            # Time quantized model  
            start_time = time.time()
            with torch.no_grad():
                _ = quantized_model(test_input)
            quantized_time = (time.time() - start_time) * 1000 / 100
            
            speedup = original_time / quantized_time
            
            print(f"   Quantization successful")
            print(f"   Original inference: {original_time:.2f}ms/sample")
            print(f"   Quantized inference: {quantized_time:.2f}ms/sample")
            print(f"   Speedup: {speedup:.2f}x")
            
            quantized_path = f'{output_dir}/mobilenet_quantized.pt'
            torch.save(quantized_model.state_dict(), quantized_path)
            
        except Exception as e:
            print(f"   Quantization failed: {e}")
        
        # 3. Create deployment package with proper JSON serialization
        deployment_info = {
            'model_name': self.model_name,
            'input_size': int(self.input_size),
            'num_classes': int(self.num_classes),
            'class_names': self.label_encoder.classes_.tolist() if self.label_encoder else None,
            'inference_time_ms': float(self.best_metrics['inference_time']),
            'accuracy_percent': float(self.best_metrics['val_acc']),
            'f1_score_percent': float(self.best_metrics['val_f1']),
            'generalization_gap_percent': float(self.best_metrics.get('generalization_gap', 0)),
            'real_time_ready': bool(self.best_metrics['inference_time'] < 50),
            'model_size_mb': float(sum(p.numel() for p in self.model.parameters()) * 4 / (1024*1024)),
            'preprocessing_steps': [
                "Normalize ECG signal to [-1, 1] range",
                "Ensure 1000 time steps (pad or truncate)",
                "Convert to torch.FloatTensor",
                "Add batch dimension if single sample"
            ],
            'deployment_files': [
                f"{self.model_name}_traced.pt",
                f"{self.model_name}_quantized.pt", 
                "deployment_info.json"
            ],
            'training_config': {
                'width_multiplier': float(self.config['width_multiplier']),
                'dropout_rate': float(self.config['dropout_rate']),
                'label_smoothing': float(self.config.get('label_smoothing', 0.0)),
                'regularization_strength': float(self.config['weight_decay'])
            }
        }
        
        with open(f'{output_dir}/deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"\nDEPLOYMENT PACKAGE READY!")
        print(f"Location: {output_dir}/")
        print(f"Mobile Ready: {'YES' if deployment_info['real_time_ready'] else 'NO'}")
        print(f"Model Size: {deployment_info['model_size_mb']:.2f} MB")
        print(f"Generalization Gap: {deployment_info['generalization_gap_percent']:.1f}%")
    
    def save_model(self, filepath):
        """Save model with enhanced metadata"""
        # Convert numpy types to Python types for JSON serialization
        class_weights_serializable = {}
        if self.class_weights:
            for k, v in self.class_weights.items():
                class_weights_serializable[int(k)] = float(v)
        
        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in self.best_metrics.items()},
            'config': self.config,
            'class_weights': class_weights_serializable,
            'label_encoder': self.label_encoder,
            'input_size': int(self.input_size),
            'num_classes': int(self.num_classes),
            'history': {k: [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v] 
                       for k, v in self.history.items()},
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'amp_used': bool(self.use_amp)
        }
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint_for_resume(self, checkpoint_path):
        """Load checkpoint to resume training"""
        if not os.path.exists(checkpoint_path):
            return 1
        
        print(f"📂 Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        if 'best_metrics' in checkpoint:
            self.best_metrics = checkpoint['best_metrics']
        
        start_epoch = checkpoint.get('best_metrics', {}).get('epoch', 0) + 1
        print(f"✅ Resuming from epoch {start_epoch}")
        return start_epoch
    
    def save_training_results(self, save_dir):
        """Save comprehensive training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert all numpy types for JSON serialization
        serializable_history = {}
        for k, v in self.history.items():
            serializable_history[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
        
        serializable_metrics = {}
        for k, v in self.best_metrics.items():
            serializable_metrics[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        
        history_file = f'{save_dir}/{self.model_name}_training_history_{timestamp}.json'
        with open(history_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'config': self.config,
                'history': serializable_history,
                'best_metrics': serializable_metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Create training plots
        self.plot_training_history(save_dir, timestamp)
        
        print(f"Results saved to {save_dir}/")
    
    def plot_training_history(self, save_dir, timestamp):
        """Create comprehensive training plots with overfitting analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss plot
        axes[0,0].plot(self.history['epochs'], self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0,0].plot(self.history['epochs'], self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0,0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Accuracy plot with overfitting indicator
        axes[0,1].plot(self.history['epochs'], self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0,1].plot(self.history['epochs'], self.history['val_acc'], 'r-', label='Validation', linewidth=2)
        axes[0,1].axhline(y=self.best_metrics['val_acc'], color='g', linestyle='--', 
                         label=f'Best: {self.best_metrics["val_acc"]:.2f}%')
        
        # Fill area showing overfitting
        axes[0,1].fill_between(self.history['epochs'], self.history['train_acc'], 
                              self.history['val_acc'], alpha=0.3, color='orange', 
                              label='Overfitting Gap')
        
        axes[0,1].set_title('Accuracy & Overfitting Gap', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy (%)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # F1 Score plot
        axes[0,2].plot(self.history['epochs'], self.history['train_f1'], 'b-', label='Train F1', linewidth=2)
        axes[0,2].plot(self.history['epochs'], self.history['val_f1'], 'r-', label='Val F1', linewidth=2)
        axes[0,2].plot(self.history['epochs'], self.history['val_balanced_acc'], 'g-', 
                      label='Val Balanced Acc', linewidth=2)
        axes[0,2].set_title('F1 Score & Balanced Accuracy', fontsize=12, fontweight='bold')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('Score (%)')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1,0].plot(self.history['epochs'], self.history['lr'], 'purple', linewidth=2)
        axes[1,0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Learning Rate')
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True, alpha=0.3)
        
        # Inference time plot
        axes[1,1].plot(self.history['epochs'], self.history['inference_time'], 'orange', linewidth=2)
        axes[1,1].axhline(y=50, color='red', linestyle='--', label='Real-time Limit (50ms)')
        axes[1,1].axhline(y=25, color='green', linestyle='--', label='Target (25ms)')
        axes[1,1].set_title('Inference Speed', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Inference Time (ms/sample)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Generalization gap over time
        generalization_gap = [train - val for train, val in zip(self.history['train_acc'], self.history['val_acc'])]
        axes[1,2].plot(self.history['epochs'], generalization_gap, 'red', linewidth=2)
        axes[1,2].axhline(y=5, color='orange', linestyle='--', label='Warning (5%)')
        axes[1,2].axhline(y=15, color='red', linestyle='--', label='Severe (15%)')
        axes[1,2].set_title('Generalization Gap (Overfitting)', fontsize=12, fontweight='bold')
        axes[1,2].set_xlabel('Epoch')
        axes[1,2].set_ylabel('Train - Val Accuracy (%)')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.suptitle(f'MobileNetV1 1D Training Analysis - {self.model_name.upper()}\n' + 
                    f'Best Validation Accuracy: {self.best_metrics["val_acc"]:.2f}% | ' +
                    f'Generalization Gap: {self.best_metrics.get("generalization_gap", 0):.1f}%', 
                    fontsize=16)
        plt.tight_layout()
        
        plot_file = f'{save_dir}/{self.model_name}_training_analysis_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to {plot_file}")

def parse_arguments():
    """Command line argument parsing for Approach 2: Generalization-First"""
    parser = argparse.ArgumentParser(description='Approach 2: Generalization-First MobileNetV1 1D ECG Training (Tight Gaps)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='mobilenet',
                       choices=['mobilenet', 'mobilenet_lite'],
                       help='Model architecture')
    
    # Training parameters - APPROACH 2 SETTINGS (Generalization-First)
    parser.add_argument('--epochs', type=int, default=120,
                       help='Number of training epochs (Approach 2: Reduced to prevent overtraining)')
    
    parser.add_argument('--batch-size', type=int, default=24,
                       help='Batch size for training (Approach 2: Smaller for better generalization)')
    
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                       help='Initial learning rate (Approach 2: Lower for stability)')
    
    parser.add_argument('--weight-decay', type=float, default=8e-4,
                       help='Weight decay for regularization (Approach 2: Higher)')
    
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                       help='Dropout rate for regularization (Approach 2: Higher)')
    
    parser.add_argument('--width-multiplier', type=float, default=0.55,
                       help='Width multiplier for MobileNet (Approach 2: Smaller)')

    parser.add_argument('--label-smoothing', type=float, default=0.15,
                       help='Label smoothing for better generalization (Approach 2: Higher)')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='balanced_ecg_smote/splits',
                       help='Directory containing preprocessed data')
    
    # Optimization parameters - APPROACH 2
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    
    parser.add_argument('--no-focal-loss', action='store_true',
                       help='Use CrossEntropy instead of Focal Loss')
    
    parser.add_argument('--gradient-clip-norm', type=float, default=0.5,
                       help='Gradient clipping norm (Approach 2: Tighter)')
    
    # Early stopping - APPROACH 2 SETTINGS (Gap-focused)
    parser.add_argument('--early-stopping-metric', type=str, default='generalization_gap',
                       choices=['val_loss', 'val_f1', 'generalization_gap'],
                       help='Metric for early stopping (Approach 2: Gap-based)')
    
    parser.add_argument('--early-stopping-patience', type=int, default=12,
                       help='Early stopping patience (Approach 2: More aggressive)')
    
    parser.add_argument('--max-generalization-gap', type=float, default=4.0,
                       help='Maximum allowed generalization gap (Approach 2)')
    
    # Runtime parameters
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with 10 epochs')
    
    parser.add_argument('--output-dir', type=str, default='outputs/models',
                       help='Output directory for models')
    
    # NEW: Approach 2 specific parameters
    parser.add_argument('--strict-gaps', action='store_true', default=True,
                       help='Enforce strict gap requirements (Approach 2)')
    
    parser.add_argument('--target-acc-gap-min', type=float, default=2.0,
                       help='Minimum target accuracy gap (Approach 2)')
    
    parser.add_argument('--target-acc-gap-max', type=float, default=5.0,
                       help='Maximum target accuracy gap (Approach 2)')
    
    return parser.parse_args()

def main():
    """Main training function with enhanced overfitting prevention"""
    args = parse_arguments()
    
    print("ENHANCED MOBILENETV1 1D ECG TRAINER")
    print("="*60)
    print("Focus: Preventing overfitting while maintaining real-time performance")
    print("Target: <50ms inference, >90% validation accuracy, <10% generalization gap")
    print("Anti-overfitting: Enhanced regularization, early stopping, generalization monitoring")
    
    # Create enhanced config with overfitting prevention
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': 10 if args.quick_test else args.epochs,
        'weight_decay': args.weight_decay,
        'dropout_rate': args.dropout_rate,
        'label_smoothing': args.label_smoothing,
        'gradient_clip_norm': args.gradient_clip_norm,
        
        # Enhanced scheduler
        'scheduler_type': 'cosine_warm_restart',
        'patience': 6,
        'factor': 0.3,
        'min_lr': 1e-9,
        'warmup_epochs': 3,
        
        # Class imbalance handling
        'use_weighted_loss': True,
        'use_focal_loss': not args.no_focal_loss,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'use_balanced_sampling': True,
        'sampling_strategy': 'weighted',
        
        # Enhanced regularization
        'use_layer_dropout': True,
        'layer_dropout_rate':0.2,
        
        # Real-time optimizations
        'use_mixed_precision': not args.no_mixed_precision,
        'gradient_accumulation_steps': 1,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_metric': args.early_stopping_metric,
        'save_best_only': True,
        
        # Model optimizations
        'width_multiplier': args.width_multiplier,
        
        # Enhanced data augmentation
        'use_augmentation': True,
        'augmentation_prob': 0.7,
        'noise_factor': 0.03,
        'time_shift_max': 20,
        'amplitude_scale_range': (0.7, 1.3),
        'time_warp_sigma': 0.3,
        'use_mixup':True,
        'mixup_alpha':0.2,
        'use_cutmix':True,
        'cutmix_alpha':0.3,

        'max_generalization_gap': getattr(args, 'max_generalization_gap', 4.0),
        'target_accuracy_gap': (getattr(args, 'target_acc_gap_min', 2.0), 
                           getattr(args, 'target_acc_gap_max', 5.0)),
        'target_f1_gap': (1.0, 4.0),
        'strict_gap_enforcement': getattr(args, 'strict_gaps', True)
    }
    
    print(f"\nTRAINING CONFIGURATION:")
    print(f"   Model: {args.model}")
    print(f"   Batch size: {config['batch_size']} (reduced for better generalization)")
    print(f"   Learning rate: {config['learning_rate']} (conservative)")
    print(f"   Weight decay: {config['weight_decay']} (increased regularization)")
    print(f"   Dropout rate: {config['dropout_rate']} (high regularization)")
    print(f"   Label smoothing: {config['label_smoothing']}")
    print(f"   Width multiplier: {config['width_multiplier']} (compact model)")
    print(f"   Early stopping: {config['early_stopping_metric']} with patience {config['early_stopping_patience']}")
    
    try:
        # Initialize trainer
        trainer = OptimizedMobileNetTrainer(args.model, config)
        
        # Load balanced data
        trainer.load_balanced_data(args.data_dir)
        
        # Setup model and training components
        trainer.setup_model()
        
        # Create optimized data loaders
        trainer.create_balanced_data_loaders()
        
        # Start training
        print(f"\nSTARTING ENHANCED TRAINING...")
        history = trainer.train(args.output_dir)
        
        # Final evaluation on test set
        best_model_path = f'{args.output_dir}/mobilenet_best.pth'
        test_results = trainer.evaluate_test_set(best_model_path)
        
        # Optimize for deployment
        trainer.optimize_for_deployment(best_model_path)
        
        # Final summary with overfitting analysis
        final_gap = trainer.best_metrics.get('generalization_gap', 0)
        
        print(f"\nTRAINING COMPLETED SUCCESSFULLY!")
        print(f"="*60)
        print(f"FINAL RESULTS:")
        print(f"   Test Accuracy: {test_results['accuracy']:.2f}%")
        print(f"   Test F1 (Weighted): {test_results['f1_weighted']:.2f}%")
        print(f"   Test F1 (Macro): {test_results['f1_macro']:.2f}%")
        print(f"   Balanced Accuracy: {test_results['balanced_accuracy']:.2f}%")
        print(f"   Generalization Gap: {final_gap:.2f}%")
        print(f"   Inference Speed: {test_results['inference_time_ms']:.2f}ms/sample")
        print(f"   Real-time Ready: {'YES' if test_results['real_time_ready'] else 'NO'}")
        
        # Overfitting assessment
        if final_gap < 3:
            print(f"\nEXCELLENT: Model shows good generalization!")
        elif final_gap < 8:
            print(f"\nGOOD: Minor overfitting, but acceptable for deployment")
        elif final_gap < 15:
            print(f"\nWARNING: Moderate overfitting detected")
            print(f"   Consider: More regularization, early stopping, or data augmentation")
        else:
            print(f"\nCRITICAL: Severe overfitting detected")
            print(f"   Recommendations:")
            print(f"   - Reduce model capacity (lower width_multiplier)")
            print(f"   - Increase regularization (higher dropout, weight_decay)")
            print(f"   - More training data or stronger augmentation")
            print(f"   - Earlier stopping or different architecture")
        
        if test_results['real_time_ready'] and final_gap < 10:
            print(f"\nSUCCESS: Model meets both real-time and generalization requirements!")
            print(f"Ready for smartphone deployment")
        elif test_results['real_time_ready']:
            print(f"\nPARTIAL SUCCESS: Real-time ready but may overfit in production")
        else:
            print(f"\nNEEDS IMPROVEMENT: Consider mobilenet_lite or width_multiplier=0.5")
        
    except FileNotFoundError as e:
        print(f"\nDATA ERROR: {e}")
        print(f"Make sure your data directory exists: {args.data_dir}")
        print(f"Expected files:")
        print(f"   - X_train.npy, y_train.npy")
        print(f"   - X_val.npy, y_val.npy") 
        print(f"   - X_test.npy, y_test.npy")
        
    except Exception as e:
        print(f"\nTRAINING ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\nTROUBLESHOOTING TIPS:")
        print(f"   1. Check CUDA availability and memory")
        print(f"   2. Reduce batch size if OOM errors")
        print(f"   3. Ensure data files are correct format")
        print(f"   4. Try --no-mixed-precision if GPU issues")
        print(f"   5. For overfitting: increase --dropout-rate or --weight-decay")
        print(f"   6. For slow convergence: adjust --learning-rate")

if __name__ == "__main__":
    main()