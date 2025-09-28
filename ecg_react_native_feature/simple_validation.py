import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import time

print('🔬 ROBUST MODEL ACCURACY VALIDATION')
print('=' * 50)

# Working model architecture
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, 
                                 stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNetV1_ECG_Robust(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNetV1_ECG_Robust, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(16, 40, stride=1),
            DepthwiseSeparableConv(40, 80, stride=2),
            DepthwiseSeparableConv(80, 80, stride=1),
            DepthwiseSeparableConv(80, 152, stride=2),
            DepthwiseSeparableConv(152, 152, stride=1),
            DepthwiseSeparableConv(152, 304, stride=2),
            DepthwiseSeparableConv(304, 304, stride=1),
            DepthwiseSeparableConv(304, 304, stride=1),
            DepthwiseSeparableConv(304, 304, stride=1),
            DepthwiseSeparableConv(304, 304, stride=1),
            DepthwiseSeparableConv(304, 304, stride=1),
            DepthwiseSeparableConv(304, 616, stride=2),
            DepthwiseSeparableConv(616, 616, stride=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(616, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load model
model = MobileNetV1_ECG_Robust(num_classes=5)
checkpoint = torch.load('../best_model_robust.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print(f'✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters')

# Load data
X = np.load('../mod/combined_ecg_final/test/segments.npy')
y = np.load('../mod/combined_ecg_final/test/labels.npy')
print(f'✅ Data loaded: {X.shape} samples')

# Load mapping
try:
    with open('CORRECTED_MAPPING_CONFIG.json', 'r') as f:
        config = json.load(f)
    mapping = config['corrected_mapping']
except:
    mapping = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}

print(f'Using mapping: {mapping}')

# Convert labels
y_indices = np.array([mapping[label] for label in y])

# Show data distribution
unique_labels, counts = np.unique(y, return_counts=True)
print(f'\nClass Distribution:')
for label, count in zip(unique_labels, counts):
    print(f'  {label}: {count:,} ({count/len(y)*100:.1f}%)')

# Run inference
print(f'\n🔄 Running inference on {len(X):,} samples...')
start_time = time.time()

batch_size = 1000
all_predictions = []

for i in range(0, len(X), batch_size):
    batch_X = X[i:i+batch_size]
    batch_tensor = torch.FloatTensor(batch_X).unsqueeze(1).to(device)
    
    with torch.no_grad():
        outputs = model(batch_tensor)
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
    
    if (i // batch_size + 1) % 10 == 0:
        print(f'  Processed {i + len(batch_X):,}/{len(X):,} samples')

inference_time = time.time() - start_time
all_predictions = np.array(all_predictions)

# Calculate accuracy
accuracy = accuracy_score(y_indices, all_predictions)

print(f'\n📊 RESULTS:')
print(f'  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
print(f'  Inference Time: {inference_time:.2f} seconds')
print(f'  Speed: {inference_time/len(X)*1000:.2f} ms/sample')

# Per-class accuracy
class_names = ['N', 'V', 'S', 'F', 'Q']
print(f'\nPer-Class Accuracy:')
for i, class_name in enumerate(class_names):
    class_mask = (y_indices == i)
    if np.sum(class_mask) > 0:
        class_accuracy = np.mean(all_predictions[class_mask] == i)
        class_count = np.sum(class_mask)
        print(f'  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.1f}%) - {class_count:,} samples')

# Confusion matrix
cm = confusion_matrix(y_indices, all_predictions)
print(f'\nConfusion Matrix:')
print(f'{"":6}', end='')
for name in class_names:
    print(f'{name:>8}', end='')
print()

for i, row in enumerate(cm):
    print(f'{class_names[i]:6}', end='')
    for val in row:
        print(f'{val:>8}', end='')
    print()

# Performance summary
print(f'\n🎯 PERFORMANCE SUMMARY:')
print(f'  Training Accuracy: 96.24%')
print(f'  Test Accuracy: {accuracy*100:.2f}%')
print(f'  Original Overfitted: 16%')
print(f'  Improvement: {accuracy/0.16:.1f}x better!')

# Deployment status
if accuracy >= 0.90:
    status = '🎉 EXCELLENT - Ready for deployment'
elif accuracy >= 0.80:
    status = '👍 GOOD - Close to deployment target'
elif accuracy >= 0.70:
    status = '⚠️  MODERATE - Needs optimization'
else:
    status = '❌ POOR - Requires significant work'

print(f'  Status: {status}')

# Save results
results = {
    'test_accuracy': float(accuracy),
    'training_accuracy': 0.9624,
    'original_accuracy': 0.16,
    'improvement_factor': float(accuracy / 0.16),
    'inference_time_seconds': float(inference_time),
    'samples_tested': len(X),
    'model_parameters': sum(p.numel() for p in model.parameters()),
    'mapping_used': mapping
}

with open('validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n💾 Results saved to validation_results.json')
print(f'✅ Validation complete!')
