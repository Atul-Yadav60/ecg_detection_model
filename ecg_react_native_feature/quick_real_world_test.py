import numpy as np
import torch
import torch.nn as nn
import time
import json

print('🌍 SIMPLIFIED REAL-WORLD TESTING')
print('=' * 40)

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

def add_noise(signal, noise_level=0.1):
    """Add simple noise to signal"""
    noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
    return signal + noise

def test_with_noise(model, device, X_test, y_test, noise_levels):
    """Test model with different noise levels"""
    
    mapping = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
    y_indices = np.array([mapping[label] for label in y_test])
    
    results = {}
    
    for noise_level in noise_levels:
        print(f'  Testing with {noise_level*100:.0f}% noise...')
        
        # Add noise to signals
        X_noisy = np.array([add_noise(signal, noise_level) for signal in X_test])
        
        # Test
        model.eval()
        correct = 0
        total = len(X_noisy)
        
        batch_size = 100
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_X = X_noisy[i:i+batch_size]
                batch_y = y_indices[i:i+batch_size]
                
                batch_tensor = torch.FloatTensor(batch_X).unsqueeze(1).to(device)
                outputs = model(batch_tensor)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                
                correct += np.sum(predictions == batch_y)
        
        accuracy = correct / total
        results[f'{noise_level*100:.0f}% noise'] = accuracy
        print(f'    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    return results

def quick_real_world_test():
    """Quick real-world testing with noise simulation"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️ Device: {device}')
    
    # Load model
    print('\n📂 Loading optimized model...')
    model = MobileNetV1_ECG_Robust(num_classes=5)
    
    try:
        checkpoint = torch.load('../best_model_focused_90.pth', map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print('✅ Model loaded successfully')
    except Exception as e:
        print(f'❌ Error loading model: {e}')
        return None
    
    # Load test data
    try:
        X = np.load('../mod/combined_ecg_final/test/segments.npy')
        y = np.load('../mod/combined_ecg_final/test/labels.npy')
        print(f'✅ Test data loaded: {len(X):,} samples')
    except Exception as e:
        print(f'❌ Error loading data: {e}')
        return None
    
    # Use subset for quick testing
    test_size = 2000
    indices = np.random.choice(len(X), test_size, replace=False)
    X_test = X[indices]
    y_test = y[indices]
    
    print(f'\n🌍 REAL-WORLD NOISE TESTING:')
    print('=' * 35)
    
    # Test with different noise levels
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    
    start_time = time.time()
    results = test_with_noise(model, device, X_test, y_test, noise_levels)
    test_time = time.time() - start_time
    
    print(f'\n📊 RESULTS SUMMARY:')
    print('=' * 20)
    
    clean_accuracy = results['0% noise']
    worst_accuracy = min(results.values())
    avg_accuracy = np.mean(list(results.values()))
    
    print(f'  Clean (0% noise): {clean_accuracy:.4f} ({clean_accuracy*100:.2f}%)')
    print(f'  Average with noise: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)')
    print(f'  Worst case: {worst_accuracy:.4f} ({worst_accuracy*100:.2f}%)')
    print(f'  Performance drop: {(clean_accuracy - worst_accuracy)*100:.2f} points')
    print(f'  Test time: {test_time:.2f}s')
    
    # Clinical readiness
    print(f'\n🏥 CLINICAL READINESS:')
    print('=' * 20)
    
    criteria = {
        'Clean ≥99%': clean_accuracy >= 0.99,
        'With noise ≥85%': avg_accuracy >= 0.85,
        'Worst case ≥70%': worst_accuracy >= 0.70
    }
    
    passed = sum(criteria.values())
    total = len(criteria)
    
    for criterion, status in criteria.items():
        print(f'  {criterion}: {"✅ PASS" if status else "❌ FAIL"}')
    
    readiness_score = passed / total * 100
    print(f'\n🎯 Readiness Score: {readiness_score:.1f}% ({passed}/{total})')
    
    if readiness_score >= 80:
        verdict = '🎉 EXCELLENT - Ready for deployment'
    elif readiness_score >= 60:
        verdict = '👍 GOOD - Suitable for controlled use'
    else:
        verdict = '⚠️ NEEDS IMPROVEMENT'
    
    print(f'🌍 Real-World Status: {verdict}')
    
    # Save results
    final_results = {
        'clean_accuracy': float(clean_accuracy),
        'average_noisy_accuracy': float(avg_accuracy),
        'worst_case_accuracy': float(worst_accuracy),
        'readiness_score': float(readiness_score),
        'all_results': {k: float(v) for k, v in results.items()},
        'deployment_ready': readiness_score >= 80,
        'test_time': float(test_time),
        'samples_tested': test_size,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('quick_real_world_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f'\n💾 Results saved to: quick_real_world_results.json')
    
    return final_results

if __name__ == "__main__":
    print('🎯 Testing optimized model with real-world noise scenarios')
    print('⏱️ Quick test - estimated time: 2-3 minutes\n')
    
    results = quick_real_world_test()
    
    if results and results['deployment_ready']:
        print(f'\n🎊 REAL-WORLD VALIDATION SUCCESS! 🎊')
        print(f'Your model maintains excellent performance under noise!')
    elif results:
        print(f'\n📊 Real-world testing complete - see results above')
    else:
        print(f'\n❌ Testing failed - check setup')
