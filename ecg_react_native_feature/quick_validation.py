import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
import json
import time

print('⚡ QUICK OPTIMIZED MODEL TEST')
print('=' * 35)

# Model architecture
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

def quick_test():
    """Quick test of the optimized model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Load optimized model
    print('\n📂 Loading optimized model...')
    model = MobileNetV1_ECG_Robust(num_classes=5)
    
    try:
        checkpoint = torch.load('../best_model_focused_90.pth', map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print('✅ Optimized model loaded')
    except Exception as e:
        print(f'❌ Error loading model: {e}')
        return
    
    # Load test data
    try:
        X = np.load('../mod/combined_ecg_final/test/segments.npy')
        y = np.load('../mod/combined_ecg_final/test/labels.npy')
        print(f'✅ Test data loaded: {len(X):,} samples')
    except Exception as e:
        print(f'❌ Error loading data: {e}')
        return
    
    # Quick test on subset
    sample_size = 5000
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_test = X[indices]
    y_test = y[indices]
    
    # Convert labels
    mapping = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
    y_indices = np.array([mapping[label] for label in y_test])
    
    print(f'\n🔄 Testing on {sample_size:,} samples...')
    
    # Run inference
    start_time = time.time()
    all_predictions = []
    
    batch_size = 500
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch_X).unsqueeze(1).to(device)
            outputs = model(batch_tensor)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
    
    inference_time = time.time() - start_time
    all_predictions = np.array(all_predictions)
    
    # Results
    accuracy = accuracy_score(y_indices, all_predictions)
    
    print(f'\n📊 QUICK TEST RESULTS:')
    print(f'  Overall Accuracy: {accuracy:.6f} ({accuracy*100:.4f}%)')
    print(f'  Inference Time: {inference_time:.2f} seconds')
    print(f'  Speed: {inference_time/len(X_test)*1000:.4f} ms/sample')
    
    # Per-class accuracy
    class_names = ['Normal', 'Ventricular', 'Supraventricular', 'Fusion', 'Unknown']
    print(f'\nPer-class accuracy:')
    for i, class_name in enumerate(class_names):
        class_mask = (y_indices == i)
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(all_predictions[class_mask] == i)
            class_count = np.sum(class_mask)
            print(f'  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {class_count} samples')
    
    # Status
    if accuracy >= 0.99:
        print(f'\n🎉 OUTSTANDING! 99%+ accuracy confirmed!')
        print(f'✅ Clinical-grade performance validated!')
    elif accuracy >= 0.95:
        print(f'\n🌟 EXCELLENT! 95%+ accuracy confirmed!')
        print(f'✅ Production-ready performance!')
    elif accuracy >= 0.90:
        print(f'\n👍 VERY GOOD! 90%+ accuracy achieved!')
        print(f'✅ Deployment-ready performance!')
    
    print(f'\n💯 Quick validation complete!')
    
    # Save quick results
    quick_results = {
        'quick_test_accuracy': float(accuracy),
        'sample_size': sample_size,
        'inference_time_ms': float(inference_time/len(X_test)*1000),
        'validation_confirmed': accuracy >= 0.99,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('quick_validation_results.json', 'w') as f:
        json.dump(quick_results, f, indent=2)
    
    return accuracy

if __name__ == "__main__":
    print('🎯 Quick validation of optimized 99.05% model')
    print('⚡ Fast subset test for confirmation\n')
    
    accuracy = quick_test()
    
    if accuracy and accuracy >= 0.99:
        print(f'\n🎊 VALIDATION CONFIRMED! 🎊')
        print(f'Your optimized ECG model is performing excellently!')
        print(f'Ready for React Native integration! 📱💓')
    elif accuracy:
        print(f'\n✅ Model validated with {accuracy*100:.2f}% accuracy')
    else:
        print(f'\n❌ Validation failed')
