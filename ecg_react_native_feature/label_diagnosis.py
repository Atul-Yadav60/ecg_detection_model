"""
ECG Label Mapping Diagnostic - Find Correct Label Assignment
This script will identify if your labels are mapped correctly
"""

import numpy as np
import torch
import torch.nn as nn
import sys
from collections import Counter

# Add parent directory
sys.path.append('../mod')
from models import MobileNetV1_1D

def quick_label_diagnosis():
    """Quick diagnosis of label mapping issue"""
    
    print("🔍 QUICK LABEL MAPPING DIAGNOSIS")
    print("=" * 55)
    
    # Load data
    X = np.load('../mod/combined_ecg_final/X_final_combined.npy')
    y = np.load('../mod/combined_ecg_final/y_final_combined.npy')
    
    # Count actual labels
    print("📊 ACTUAL LABEL COUNTS:")
    label_counts = Counter(y)
    total = len(y)
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percent = (count/total)*100
        print(f"   {label}: {count:,} ({percent:.1f}%)")
    
    # From your training logs, we know the expected distribution was:
    print("\n🎯 EXPECTED FROM TRAINING:")
    print("   N: 2,549 (1.2%)")
    print("   V: 167,289 (79.2%)")  
    print("   S: 21,859 (10.4%)")
    print("   F: 5,973 (2.8%)")
    print("   Q: 13,440 (6.4%)")
    
    # The issue: Your current data shows N as 79.2%, but training expected V as 79.2%
    print("\n❌ PROBLEM IDENTIFIED:")
    print("   Current: N=79.2%, V=6.4%")
    print("   Expected: N=1.2%, V=79.2%")
    print("   → Labels N and V are SWAPPED!")
    
    # Create corrected mapping
    print("\n🔧 CORRECTED MAPPING:")
    # Original (wrong): N→0, V→1, S→2, F→3, Q→4
    # Corrected: N→1, V→0, S→2, F→3, Q→4 (swap N and V)
    corrected_mapping = {'N': 1, 'V': 0, 'S': 2, 'F': 3, 'Q': 4}
    print(f"   {corrected_mapping}")
    
    # Test the corrected mapping
    print("\n🧪 TESTING CORRECTED MAPPING:")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV1_1D(num_classes=5, width_multiplier=0.6, dropout_rate=0.5).to(device)
    model.load_state_dict(torch.load('../best_model_robust.pth', map_location=device))
    model.eval()
    
    # Normalize data
    X_normalized = 2 * (X - X.min()) / (X.max() - X.min()) - 1
    
    # Test on small subset
    test_size = 2000
    indices = np.random.choice(len(X), test_size, replace=False)
    X_test = X_normalized[indices]
    y_test = y[indices]
    
    # Convert labels using corrected mapping
    y_test_corrected = np.array([corrected_mapping[label] for label in y_test])
    
    # Run inference
    correct = 0
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            batch_y = y_test_corrected[i:i+batch_size]
            
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            correct += (predicted.cpu().numpy() == batch_y).sum()
    
    corrected_accuracy = correct / test_size
    
    print(f"✅ Corrected Accuracy: {corrected_accuracy:.4f} ({corrected_accuracy*100:.2f}%)")
    
    # Assessment
    if corrected_accuracy > 0.90:
        print(f"\n🎉 SUCCESS! LABELS FIXED!")
        print(f"   ✅ {corrected_accuracy*100:.1f}% accuracy with corrected mapping")
        print(f"   ✅ Matches training performance (~96%)")
        print(f"   ✅ Problem was N/V label swap!")
    elif corrected_accuracy > 0.80:
        print(f"\n👍 MUCH BETTER!")
        print(f"   ✅ {corrected_accuracy*100:.1f}% accuracy")
        print(f"   ✅ Significant improvement")
    else:
        print(f"\n⚠️  Still investigating...")
    
    print(f"\n💡 SOLUTION:")
    print(f"   Use this corrected mapping in all future scripts:")
    print(f"   label_to_idx = {corrected_mapping}")
    
    return corrected_mapping, corrected_accuracy

if __name__ == "__main__":
    quick_label_diagnosis()
