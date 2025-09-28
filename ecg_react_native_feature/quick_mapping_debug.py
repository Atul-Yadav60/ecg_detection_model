"""
Quick Label Mapping Debug - Find the correct mapping
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from collections import Counter

class MobileNetV1_1D(nn.Module):
    def __init__(self, num_classes=5, width_multiplier=0.6):
        super(MobileNetV1_1D, self).__init__()
        
        def make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        input_channel = make_divisible(32 * width_multiplier, 8)
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv1d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm1d(inp),
                nn.ReLU(inplace=True),
                
                nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU(inplace=True)
            )
        
        self.model = nn.Sequential(
            conv_bn(1, input_channel, 2),
            conv_dw(input_channel, make_divisible(64 * width_multiplier, 8), 1),
            conv_dw(make_divisible(64 * width_multiplier, 8), make_divisible(128 * width_multiplier, 8), 2),
            conv_dw(make_divisible(128 * width_multiplier, 8), make_divisible(128 * width_multiplier, 8), 1),
            conv_dw(make_divisible(128 * width_multiplier, 8), make_divisible(256 * width_multiplier, 8), 2),
            conv_dw(make_divisible(256 * width_multiplier, 8), make_divisible(256 * width_multiplier, 8), 1),
            conv_dw(make_divisible(256 * width_multiplier, 8), make_divisible(512 * width_multiplier, 8), 2),
            conv_dw(make_divisible(512 * width_multiplier, 8), make_divisible(512 * width_multiplier, 8), 1),
            conv_dw(make_divisible(512 * width_multiplier, 8), make_divisible(512 * width_multiplier, 8), 1),
            conv_dw(make_divisible(512 * width_multiplier, 8), make_divisible(512 * width_multiplier, 8), 1),
            conv_dw(make_divisible(512 * width_multiplier, 8), make_divisible(512 * width_multiplier, 8), 1),
            conv_dw(make_divisible(512 * width_multiplier, 8), make_divisible(512 * width_multiplier, 8), 1),
            conv_dw(make_divisible(512 * width_multiplier, 8), make_divisible(1024 * width_multiplier, 8), 2),
            conv_dw(make_divisible(1024 * width_multiplier, 8), make_divisible(1024 * width_multiplier, 8), 1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Linear(make_divisible(1024 * width_multiplier, 8), num_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def quick_mapping_test():
    print("🔍 QUICK LABEL MAPPING TEST")
    print("=" * 40)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV1_1D(num_classes=5, width_multiplier=0.6)
    
    try:
        checkpoint = torch.load('../best_model_robust.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Load test data
    try:
        X_test = np.load('../mod/combined_ecg_final/test/X_test.npy')
        y_test = np.load('../mod/combined_ecg_final/test/y_test.npy')
        print(f"✅ Test data: {X_test.shape}, {y_test.shape}")
    except:
        print("❌ Could not load test data, using balanced data")
        X_test = np.load('../mod/balanced_ecg_smote/X_balanced.npy')[:1000]  # Sample
        y_test = np.load('../mod/balanced_ecg_smote/y_balanced.npy')[:1000]
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    # Test different mappings
    mappings = {
        'Original': {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4},
        'N-V Swap': {'N': 1, 'V': 0, 'S': 2, 'F': 3, 'Q': 4},
        'V-N Swap': {'V': 0, 'N': 1, 'S': 2, 'F': 3, 'Q': 4}  # Alternative notation
    }
    
    print("\n📊 TESTING MAPPINGS:")
    best_acc = 0
    best_mapping = None
    
    for name, mapping in mappings.items():
        # Convert labels using this mapping
        y_mapped = np.array([mapping[label] for label in y_test])
        accuracy = accuracy_score(y_mapped, predictions)
        
        print(f"  {name:12}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_mapping = (name, mapping)
    
    if best_mapping:
        print(f"\n🏆 BEST: {best_mapping[0]} with {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"   Mapping: {best_mapping[1]}")
    else:
        print(f"\n🏆 BEST: No mapping found with accuracy > 0")
    
    # Show label distribution
    print(f"\n📈 LABEL DISTRIBUTION:")
    label_counts = Counter(y_test)
    total = len(y_test)
    for label in ['N', 'V', 'S', 'F', 'Q']:
        count = label_counts.get(label, 0)
        pct = (count / total) * 100
        print(f"  {label}: {count:4} ({pct:5.1f}%)")
    
    return best_mapping

if __name__ == "__main__":
    quick_mapping_test()
