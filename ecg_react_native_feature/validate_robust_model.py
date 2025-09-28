import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

print('🔬 COMPREHENSIVE ROBUST MODEL VALIDATION')
print('=' * 60)

# WORKING MODEL ARCHITECTURE (exactly matches the saved robust checkpoint)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, 
                                 stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        # Pointwise convolution  
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
        
        # Features extraction - matches exact checkpoint structure
        self.features = nn.Sequential(
            # Initial conv: 1 → 16 channels (features.0, features.1, features.2)
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # Depthwise separable convolutions (features.3 through features.15)
            DepthwiseSeparableConv(16, 40, stride=1),    # features.3: 16→40
            DepthwiseSeparableConv(40, 80, stride=2),    # features.4: 40→80
            DepthwiseSeparableConv(80, 80, stride=1),    # features.5: 80→80
            DepthwiseSeparableConv(80, 152, stride=2),   # features.6: 80→152
            DepthwiseSeparableConv(152, 152, stride=1),  # features.7: 152→152
            DepthwiseSeparableConv(152, 304, stride=2),  # features.8: 152→304
            DepthwiseSeparableConv(304, 304, stride=1),  # features.9: 304→304
            DepthwiseSeparableConv(304, 304, stride=1),  # features.10: 304→304
            DepthwiseSeparableConv(304, 304, stride=1),  # features.11: 304→304
            DepthwiseSeparableConv(304, 304, stride=1),  # features.12: 304→304
            DepthwiseSeparableConv(304, 304, stride=1),  # features.13: 304→304
            DepthwiseSeparableConv(304, 616, stride=2),  # features.14: 304→616
            DepthwiseSeparableConv(616, 616, stride=1),  # features.15: 616→616
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classifier (matches checkpoint: classifier.0 (dropout), classifier.1 (linear))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(616, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Setup device and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔧 Device: {device}')

model = MobileNetV1_ECG_Robust(num_classes=5)
print(f'📊 Model created with {sum(p.numel() for p in model.parameters()):,} parameters')

# Load the robust model
print(f'\n🔄 Loading robust model checkpoint...')
try:
    checkpoint = torch.load('../best_model_robust.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print('✅ Model loaded successfully!')
except Exception as e:
    print(f'❌ Error loading model: {e}')
    exit()

# Load test datasets
datasets_to_test = [
    ('../mod/combined_ecg_final/test/segments.npy', '../mod/combined_ecg_final/test/labels.npy', 'Final Test Data'),
    ('../mod/balanced_ecg_smote/X_balanced.npy', '../mod/balanced_ecg_smote/y_balanced.npy', 'Balanced SMOTE Data')
]

# Load corrected mapping
try:
    with open('CORRECTED_MAPPING_CONFIG.json', 'r') as f:
        config = json.load(f)
    mapping = config['corrected_mapping']
    reverse_mapping = config['reverse_mapping']
    print(f'✅ Using corrected mapping: {mapping}')
except:
    mapping = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
    reverse_mapping = {0: 'N', 1: 'V', 2: 'S', 3: 'F', 4: 'Q'}
    print(f'⚠️  Using original mapping: {mapping}')

class_names = ['Normal (N)', 'Ventricular (V)', 'Supraventricular (S)', 'Fusion (F)', 'Unknown (Q)']

# Test on each dataset
validation_results = {}

for X_path, y_path, dataset_name in datasets_to_test:
    print(f'\n{"="*60}')
    print(f'🧪 TESTING ON: {dataset_name}')
    print(f'{"="*60}')
    
    try:
        # Load data
        X = np.load(X_path)
        y = np.load(y_path)
        print(f'📊 Dataset loaded: {X.shape} samples')
        print(f'   Data range: {X.min():.3f} to {X.max():.3f}')
        
        # Show class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f'   Class distribution:')
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(y)) * 100
            print(f'     {label}: {count:,} ({percentage:.1f}%)')
        
        # Convert labels to indices
        y_indices = np.array([mapping[label] for label in y])
        
        # Run inference in batches
        print(f'\n🔄 Running inference...')
        batch_size = 500
        all_predictions = []
        all_probabilities = []
        inference_times = []
        
        start_time = time.time()
        
        for i in range(0, len(X), batch_size):
            batch_start = time.time()
            
            batch_X = X[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch_X).unsqueeze(1).to(device)
            
            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
            
            batch_time = time.time() - batch_start
            inference_times.append(batch_time)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f'   Processed {i + len(batch_X):,}/{len(X):,} samples...')
        
        total_time = time.time() - start_time
        avg_inference_time = total_time / len(X) * 1000  # ms per sample
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(y_indices, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(y_indices, all_predictions, average=None)
        
        # Ensure arrays for consistent handling
        precision = np.atleast_1d(precision) if precision is not None else np.array([0.0])
        recall = np.atleast_1d(recall) if recall is not None else np.array([0.0])
        f1 = np.atleast_1d(f1) if f1 is not None else np.array([0.0])
        support = np.atleast_1d(support) if support is not None else np.array([0])
        
        macro_f1 = np.mean(f1)
        weighted_f1 = np.average(f1, weights=support)
        
        print(f'\n📈 RESULTS FOR {dataset_name}:')
        print(f'   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
        print(f'   Macro F1-Score: {macro_f1:.4f}')
        print(f'   Weighted F1-Score: {weighted_f1:.4f}')
        print(f'   Inference Speed: {avg_inference_time:.2f} ms/sample')
        print(f'   Total Time: {total_time:.2f} seconds')
        
        # Per-class metrics
        print(f'\n📊 PER-CLASS PERFORMANCE:')
        print(f'{"Class":15} {"Precision":>10} {"Recall":>10} {"F1-Score":>10} {"Support":>10}')
        print('-' * 65)
        
        for i, (class_name, prec, rec, f1_score, sup) in enumerate(zip(class_names, precision, recall, f1, support)):
            print(f'{class_name:15} {prec:>10.4f} {rec:>10.4f} {f1_score:>10.4f} {sup:>10,}')
        
        # Confusion Matrix
        cm = confusion_matrix(y_indices, all_predictions)
        print(f'\n📋 CONFUSION MATRIX:')
        print(f'{"":8}', end='')
        for name in [reverse_mapping[i] for i in range(5)]:
            print(f'{name:>8}', end='')
        print()
        
        for i, row in enumerate(cm):
            print(f'{reverse_mapping[i]:8}', end='')
            for val in row:
                print(f'{val:>8}', end='')
            print()
        
        # Save results
        validation_results[dataset_name] = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'inference_time_ms': float(avg_inference_time),
            'total_samples': len(X)
        }
        
        # Confidence analysis
        max_probs = np.max(all_probabilities, axis=1)
        correct_predictions = (all_predictions == y_indices)
        
        print(f'\n🎯 CONFIDENCE ANALYSIS:')
        print(f'   Average Confidence: {np.mean(max_probs):.4f}')
        print(f'   Confidence on Correct: {np.mean(max_probs[correct_predictions]):.4f}')
        print(f'   Confidence on Incorrect: {np.mean(max_probs[~correct_predictions]):.4f}')
        
        # Low confidence predictions
        low_conf_threshold = 0.7
        low_conf_mask = max_probs < low_conf_threshold
        print(f'   Low Confidence (<{low_conf_threshold}): {np.sum(low_conf_mask):,} ({np.mean(low_conf_mask)*100:.1f}%)')
        
        if np.sum(low_conf_mask) > 0:
            low_conf_accuracy = np.mean(correct_predictions[low_conf_mask])
            print(f'   Accuracy on Low Confidence: {low_conf_accuracy:.4f} ({low_conf_accuracy*100:.1f}%)')
        
    except Exception as e:
        print(f'❌ Error testing {dataset_name}: {e}')
        continue

# Overall summary
print(f'\n{"="*60}')
print(f'🎯 OVERALL VALIDATION SUMMARY')
print(f'{"="*60}')

print(f'Model Architecture: MobileNetV1_ECG_Robust')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Training Accuracy: 96.24%')
print(f'Label Mapping: {mapping}')

for dataset_name, results in validation_results.items():
    print(f'\n📊 {dataset_name}:')
    print(f'   Accuracy: {results["accuracy"]:.4f} ({results["accuracy"]*100:.2f}%)')
    print(f'   Macro F1: {results["macro_f1"]:.4f}')
    print(f'   Samples: {results["total_samples"]:,}')
    print(f'   Speed: {results["inference_time_ms"]:.2f} ms/sample')

# Performance improvement analysis
print(f'\n🚀 PERFORMANCE IMPROVEMENT:')
print(f'   Original Model: 16% accuracy (overfitted)')
print(f'   Robust Model Training: 96.24% accuracy')

for dataset_name, results in validation_results.items():
    improvement = results["accuracy"] / 0.16
    print(f'   {dataset_name}: {results["accuracy"]*100:.2f}% ({improvement:.1f}x improvement)')

# Deployment readiness
print(f'\n🎯 DEPLOYMENT READINESS:')
best_accuracy = max([results["accuracy"] for results in validation_results.values()])

if best_accuracy >= 0.90:
    status = "🎉 READY FOR DEPLOYMENT"
elif best_accuracy >= 0.80:
    status = "👍 GOOD - CLOSE TO DEPLOYMENT"
elif best_accuracy >= 0.70:
    status = "⚠️  MODERATE - NEEDS OPTIMIZATION"
else:
    status = "❌ POOR - REQUIRES SIGNIFICANT WORK"

print(f'   Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)')
print(f'   Status: {status}')

# Save comprehensive results
results_summary = {
    'model_info': {
        'architecture': 'MobileNetV1_ECG_Robust',
        'parameters': sum(p.numel() for p in model.parameters()),
        'training_accuracy': 0.9624,
        'label_mapping': mapping
    },
    'validation_results': validation_results,
    'best_accuracy': float(best_accuracy),
    'improvement_factor': float(best_accuracy / 0.16),
    'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

with open('robust_model_validation_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f'\n💾 Results saved to: robust_model_validation_results.json')
print(f'✅ Validation complete!')
