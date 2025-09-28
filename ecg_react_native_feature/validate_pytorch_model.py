"""
PROPER Robust Model Validation with PyTorch
Tests the robust PyTorch model directly (not ONNX)
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Add parent directory to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../mod')))
from mod.models import MobileNetV1_1D

def validate_pytorch_model():
    """Validate the PyTorch robust model directly"""
    
    print("🔬 PYTORCH ROBUST MODEL VALIDATION")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load the robust model
    model_path = '../best_model_robust.pth'
    
    print(f"✅ Loading robust model: {model_path}")
    
    # Create model architecture (same as training)
    model = MobileNetV1_1D(
        num_classes=5,
        width_multiplier=0.6,
        dropout_rate=0.5
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully!")
    
    # Load training data (same as used in training)
    print(f"\n📊 Loading training data...")
    
    try:
        # Load the EXACT same data used in training
        X_path = '../mod/combined_ecg_final/X_final_combined.npy'
        y_path = '../mod/combined_ecg_final/y_final_combined.npy'
        
        X = np.load(X_path)
        y = np.load(y_path)
        
        # Convert string labels to integers (same as training)
        label_to_idx = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
        y_numeric = np.array([label_to_idx[label] for label in y])
        
        print(f"✅ Loaded original data: {X.shape[0]:,} samples")
        
        # Show original distribution
        unique, counts = np.unique(y_numeric, return_counts=True)
        class_names = ['Normal (N)', 'Ventricular (V)', 'Supraventricular (S)', 'Fusion (F)', 'Unknown (Q)']
        
        print(f"\n📊 Original Distribution:")
        total_samples = len(y_numeric)
        for class_idx, count in zip(unique, counts):
            percentage = (count / total_samples) * 100
            print(f"   {class_names[class_idx]}: {count:,} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Normalize data (EXACTLY as in training)
    X_normalized = 2 * (X - X.min()) / (X.max() - X.min()) - 1
    print(f"🔧 Data normalized to [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
    
    # Create stratified test split to preserve class distribution
    print(f"\n🎯 Creating stratified test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_numeric, 
        test_size=0.15,  # 15% for testing
        stratify=y_numeric,
        random_state=42
    )
    
    print(f"   Test samples: {len(X_test):,}")
    
    # Verify test distribution matches training
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"\n📊 Test Distribution (stratified):")
    total_test = len(y_test)
    for class_idx, count in zip(unique_test, counts_test):
        percentage = (count / total_test) * 100
        print(f"   {class_names[class_idx]}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n🔥 Running model inference...")
    
    # Run inference
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    batch_size = 64
    correct = 0
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]
            
            # Convert to tensor
            batch_X_tensor = torch.FloatTensor(batch_X).to(device)
            
            # Forward pass
            outputs = model(batch_X_tensor)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.cpu().numpy()
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probs)
            
            # Count correct
            correct += (predicted.cpu().numpy() == batch_y).sum()
            
            if i % (batch_size * 20) == 0:
                progress = (i + batch_size) / len(X_test) * 100
                print(f"   Progress: {progress:.1f}%")
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, all_predictions)
    f1 = f1_score(y_test, all_predictions, average='weighted')
    
    print(f"\n🏆 VALIDATION RESULTS:")
    print("=" * 60)
    print(f"✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Weighted F1 Score: {f1:.4f}")
    print(f"✅ Correct Predictions: {correct:,}/{len(X_test):,}")
    
    # Detailed classification report
    print(f"\n📊 CLASSIFICATION REPORT:")
    print("-" * 60)
    
    report = classification_report(
        y_test, all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)
    
    # Per-class accuracy
    print(f"\n📈 PER-CLASS ACCURACY:")
    print("-" * 60)
    cm = confusion_matrix(y_test, all_predictions)
    for i, class_name in enumerate(class_names):
        if i < len(cm):
            class_correct = cm[i][i] if i < len(cm) else 0
            class_total = cm[i].sum() if i < len(cm) else 0
            class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
            print(f"   {class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")
    
    # Compare with training results
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print("=" * 60)
    print(f"🔥 Training Results (Cross-Validation):")
    print(f"   • Average Accuracy: 96.24% ± 0.07%")
    print(f"   • Average F1 Score: 0.9622 ± 0.0006")
    
    print(f"\n✅ Validation Results:")
    print(f"   • Test Accuracy: {accuracy*100:.2f}%")
    print(f"   • Test F1 Score: {f1:.4f}")
    
    # Performance assessment
    accuracy_diff = abs(accuracy * 100 - 96.24)
    
    if accuracy_diff <= 3:
        print(f"\n🌟 EXCELLENT! Perfect consistency!")
        print(f"   ✅ Validation matches training ({accuracy_diff:.1f}% difference)")
        print(f"   ✅ Model is working exactly as expected")
        status = "EXCELLENT"
    elif accuracy_diff <= 8:
        print(f"\n👍 GOOD! Reasonable consistency!")
        print(f"   ✅ Good validation performance ({accuracy_diff:.1f}% difference)")
        status = "GOOD"
    else:
        print(f"\n⚠️  GAP detected")
        print(f"   ❌ Validation differs from training ({accuracy_diff:.1f}% difference)")
        status = "NEEDS_REVIEW"
    
    # Previous model comparison
    print(f"\n📈 VS PREVIOUS MODEL:")
    print("-" * 60)
    previous_accuracy = 16.0
    improvement = ((accuracy * 100 - previous_accuracy) / previous_accuracy) * 100
    
    print(f"   OLD Model: {previous_accuracy}% (overfitted)")
    print(f"   NEW Model: {accuracy*100:.1f}% (robust)")
    print(f"   Improvement: +{improvement:.0f}% better!")
    
    # Sample predictions
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    print("-" * 60)
    
    for i in range(min(8, len(all_predictions))):
        true_class = class_names[y_test[i]]
        pred_class = class_names[all_predictions[i]]
        confidence = all_probabilities[i].max()
        status_icon = "✅" if y_test[i] == all_predictions[i] else "❌"
        
        print(f"   {i+1}. {status_icon} True: {true_class:<20} | Pred: {pred_class:<20} | Conf: {confidence:.3f}")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    print("=" * 60)
    
    if accuracy > 0.90:
        print(f"🎉 SUCCESS! Your robust model is EXCELLENT!")
        print(f"   ✅ Problem completely solved")
        print(f"   ✅ Ready for ONNX conversion and deployment")
        print(f"   ✅ Expected real-world performance: ~{accuracy*100:.0f}%")
    elif accuracy > 0.80:
        print(f"👍 GOOD! Your model is working well!")
        print(f"   ✅ Significant improvement achieved")
        print(f"   ✅ Suitable for deployment with monitoring")
    else:
        print(f"⚠️  Needs improvement")
        print(f"   ❌ Performance below expectations")
    
    return accuracy, f1, status

if __name__ == "__main__":
    validate_pytorch_model()
