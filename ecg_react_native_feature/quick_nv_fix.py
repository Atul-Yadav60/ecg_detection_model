"""
Quick N/V Mapping Fix - Focused Solution
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

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

print("🔧 QUICK N/V MAPPING FIX")
print("=" * 30)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileNetV1_1D(num_classes=5, width_multiplier=0.6)

print(f"Device: {device}")

try:
    checkpoint = torch.load('../best_model_robust.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model error: {e}")
    exit()

# Load test data - try both locations
try:
    X_test = np.load('../mod/combined_ecg_final/test/segments.npy')
    y_test = np.load('../mod/combined_ecg_final/test/labels.npy')
    print("✅ Using final test data")
except:
    try:
        # Use smaller sample from balanced data
        X_all = np.load('../mod/balanced_ecg_smote/X_balanced.npy')
        y_all = np.load('../mod/balanced_ecg_smote/y_balanced.npy')
        
        # Take a sample to avoid memory issues
        indices = np.random.choice(len(X_all), min(5000, len(X_all)), replace=False)
        X_test = X_all[indices]
        y_test = y_all[indices]
        print("✅ Using balanced sample data")
    except Exception as e:
        print(f"❌ Data error: {e}")
        exit()

print(f"Data shape: {X_test.shape}, {y_test.shape}")

# Quick inference
print("🧠 Getting predictions...")
X_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)

with torch.no_grad():
    outputs = model(X_tensor)
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

print("✅ Predictions complete")

# Test the key mappings
mappings = {
    'Original': {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4},
    'N-V Swap': {'N': 1, 'V': 0, 'S': 2, 'F': 3, 'Q': 4}
}

print("\n🧪 Testing mappings:")
for name, mapping in mappings.items():
    try:
        y_mapped = np.array([mapping[label] for label in y_test])
        accuracy = accuracy_score(y_mapped, predictions)
        print(f"  {name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    except Exception as e:
        print(f"  {name}: Error - {e}")

print("\n✅ Quick test complete!")
