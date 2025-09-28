import numpy as np
import torch
import torch.nn as nn
import json
import time
import os

print('📊 TRAINING PROGRESS MONITOR')
print('=' * 30)

# Model architecture (same as training)
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

def quick_test_current_model():
    """Quick test of currently available models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check available models
    model_files = []
    base_path = '..'
    for file in os.listdir(base_path):
        if file.endswith('.pth'):
            model_files.append(file)
    
    print(f'📂 Available models:')
    for model_file in model_files:
        print(f'  - {model_file}')
    
    # Test the best available model
    best_model_file = None
    if 'best_model_optimized_90plus.pth' in model_files:
        best_model_file = 'best_model_optimized_90plus.pth'
        print(f'\n🎯 Testing optimized 90%+ model...')
    elif 'best_model_robust.pth' in model_files:
        best_model_file = 'best_model_robust.pth'
        print(f'\n🔧 Testing baseline robust model...')
    else:
        print(f'❌ No suitable model found')
        return None
    
    # Load model
    model = MobileNetV1_ECG_Robust(num_classes=5)
    try:
        checkpoint = torch.load(f'../{best_model_file}', map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f'✅ Loaded {best_model_file}')
    except Exception as e:
        print(f'❌ Error loading {best_model_file}: {e}')
        return None
    
    # Load test data
    try:
        X = np.load('../mod/combined_ecg_final/test/segments.npy')
        y = np.load('../mod/combined_ecg_final/test/labels.npy')
        print(f'✅ Loaded {len(X):,} test samples')
    except Exception as e:
        print(f'❌ Error loading data: {e}')
        return None
    
    # Quick accuracy test
    mapping = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
    y_indices = np.array([mapping[label] for label in y])
    
    # Test on subset for speed
    sample_size = min(3000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_test = X[indices]
    y_test = y_indices[indices]
    
    print(f'🔄 Testing on {sample_size:,} samples...')
    
    all_predictions = []
    batch_size = 200
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch_X).unsqueeze(1).to(device)
            
            outputs = model(batch_tensor)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
    
    test_time = time.time() - start_time
    all_predictions = np.array(all_predictions)
    
    # Calculate accuracy
    accuracy = np.mean(all_predictions == y_test)
    
    print(f'\n📊 CURRENT MODEL PERFORMANCE:')
    print(f'  Model: {best_model_file}')
    print(f'  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'  Test Time: {test_time:.2f} seconds')
    print(f'  Speed: {test_time/len(X_test)*1000:.2f} ms/sample')
    
    # Per-class breakdown
    class_names = ['N', 'V', 'S', 'F', 'Q']
    print(f'\nPer-class accuracy:')
    for i, class_name in enumerate(class_names):
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(all_predictions[class_mask] == i)
            class_count = np.sum(class_mask)
            print(f'  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.1f}%) - {class_count} samples')
    
    # Status
    if accuracy >= 0.90:
        status = '🎉 EXCELLENT! 90%+ accuracy achieved!'
        emoji = '🚀'
    elif accuracy >= 0.85:
        status = '👍 VERY GOOD! Close to 90% target'
        emoji = '⭐'
    elif accuracy >= 0.80:
        status = '👌 GOOD! Solid improvement'
        emoji = '💪'
    else:
        status = '⚠️ MODERATE - Needs more optimization'
        emoji = '🔧'
    
    print(f'\n{emoji} STATUS: {status}')
    
    # Save quick test results
    results = {
        'model_file': best_model_file,
        'test_accuracy': float(accuracy),
        'test_samples': sample_size,
        'test_time_ms': float(test_time / len(X_test) * 1000),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'target_achieved': accuracy >= 0.90
    }
    
    with open('quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return accuracy

def monitor_training_progress():
    """Monitor if training is producing results"""
    print(f'\n🔍 MONITORING TRAINING FILES...')
    
    # Check for result files
    result_files = [
        'optimized_model_results.json',
        'quick_test_results.json',
        'validation_results.json'
    ]
    
    found_results = []
    for result_file in result_files:
        if os.path.exists(result_file):
            found_results.append(result_file)
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                print(f'📄 {result_file}:')
                for key, value in data.items():
                    if 'accuracy' in key.lower():
                        if isinstance(value, float):
                            print(f'  {key}: {value:.4f} ({value*100:.2f}%)')
                        else:
                            print(f'  {key}: {value}')
                print()
            except:
                print(f'  {result_file}: Found but could not read')
    
    if not found_results:
        print('📝 No result files found yet - training may still be in progress')
    
    # Check for model files
    model_files = []
    base_path = '..'
    for file in os.listdir(base_path):
        if file.endswith('.pth') and 'optimized' in file:
            model_files.append(file)
    
    if model_files:
        print(f'🎯 Found optimized models:')
        for model_file in model_files:
            print(f'  - {model_file}')
    
    return len(found_results), len(model_files)

if __name__ == "__main__":
    print('🎯 GOAL: Monitor progress toward 90%+ accuracy')
    print('⏱️  Running quick performance check...\n')
    
    # Monitor training progress
    result_count, model_count = monitor_training_progress()
    
    # Quick test current best model
    current_accuracy = quick_test_current_model()
    
    print(f'\n📋 SUMMARY:')
    print(f'  Result files found: {result_count}')
    print(f'  Optimized models: {model_count}')
    if current_accuracy:
        print(f'  Current accuracy: {current_accuracy*100:.2f}%')
        if current_accuracy >= 0.90:
            print(f'  🎉 TARGET ACHIEVED! Ready for deployment!')
        else:
            remaining = 0.90 - current_accuracy
            print(f'  🎯 Need +{remaining*100:.2f} points to reach 90%')
    
    print(f'\n✅ Monitoring complete!')
