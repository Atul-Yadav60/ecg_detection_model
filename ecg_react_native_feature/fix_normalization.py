import numpy as np
import onnxruntime as ort
from sklearn.preprocessing import RobustScaler
from collections import defaultdict

def normalize_ecg_signal(signal, method='robust'):
    """Apply the same normalization as in training"""
    signal = np.array(signal).reshape(-1, 1)
    
    if method == 'robust':
        scaler = RobustScaler()
        normalized = scaler.fit_transform(signal).flatten()
    elif method == 'zscore':
        normalized = (signal - np.mean(signal)) / np.std(signal)
        normalized = normalized.flatten()
    elif method == 'minmax':
        normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        normalized = normalized.flatten()
    elif method == 'minmax_to_range':
        # Normalize to [-1, 1] range as mentioned in training logs
        normalized = 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
        normalized = normalized.flatten()
    else:
        # Default to robust scaling if method is not recognized
        scaler = RobustScaler()
        normalized = scaler.fit_transform(signal).flatten()
    
    return normalized

def test_all_normalization_methods():
    """Test which normalization method gives best results"""
    
    print("🧪 TESTING DIFFERENT NORMALIZATION METHODS")
    print("=" * 60)
    
    # Load model
    session = ort.InferenceSession('mobilenet_v1_ecg_model.onnx')
    
    # Load test data (small sample for speed)
    X = np.load('../mod/balanced_ecg_smote/X_balanced.npy')
    y = np.load('../mod/balanced_ecg_smote/y_balanced.npy')
    
    class_names = ['Normal (N)', 'Ventricular (V)', 'Supraventricular (S)', 'Fusion (F)', 'Unknown (Q)']
    label_to_idx = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
    
    # Test different normalization methods
    normalization_methods = ['robust', 'zscore', 'minmax', 'minmax_to_range', 'none']
    
    # Test on 100 samples
    test_size = 100
    indices = np.random.choice(len(X), test_size, replace=False)
    
    results = {}
    
    for norm_method in normalization_methods:
        print(f"\n📊 Testing normalization: {norm_method}")
        
        correct = 0
        predictions_made = 0
        
        for i, idx in enumerate(indices):
            try:
                # Get original sample
                original_sample = X[idx].copy()
                true_label_str = y[idx]
                true_idx = label_to_idx[true_label_str]
                
                # Apply normalization
                if norm_method == 'none':
                    test_sample = original_sample.reshape(1, -1).astype(np.float32)
                else:
                    normalized_sample = normalize_ecg_signal(original_sample, method=norm_method)
                    test_sample = normalized_sample.reshape(1, -1).astype(np.float32)
                
                # Run inference
                outputs = session.run(None, {'ecg_input': test_sample})
                raw_preds = outputs[0][0]
                pred_idx = np.argmax(raw_preds)
                
                if pred_idx == true_idx:
                    correct += 1
                predictions_made += 1
                
            except Exception as e:
                print(f"Error with sample {i}: {e}")
                continue
        
        accuracy = correct / predictions_made * 100 if predictions_made > 0 else 0
        results[norm_method] = accuracy
        print(f"   Accuracy: {accuracy:.1f}% ({correct}/{predictions_made})")
    
    # Find best method
    print(f"\n🏆 NORMALIZATION RESULTS:")
    print("=" * 60)
    best_method = max(results, key=lambda x: results[x])
    
    for method, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "🏆" if method == best_method else "  "
        print(f"{marker} {method:>15}: {accuracy:>5.1f}%")
    
    print(f"\n✅ Best normalization method: {best_method} ({results[best_method]:.1f}%)")
    
    # Test best method on more samples
    if results[best_method] > 50:  # Only if reasonable accuracy
        print(f"\n🔍 Detailed test with {best_method} normalization:")
        test_with_best_normalization(best_method, session, X, y, class_names, label_to_idx)
    
    return best_method

def test_with_best_normalization(norm_method, session, X, y, class_names, label_to_idx):
    """Test model with the best normalization method"""
    
    # Test on 300 samples for better statistics
    test_size = 300
    indices = np.random.choice(len(X), test_size, replace=False)
    
    correct_predictions = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    confusion_matrix = np.zeros((5, 5), dtype=int)
    
    print(f"   Testing on {test_size} samples...")
    
    for i, idx in enumerate(indices):
        try:
            # Prepare input with best normalization
            original_sample = X[idx].copy()
            true_label_str = y[idx]
            true_idx = label_to_idx[true_label_str]
            
            if norm_method == 'none':
                test_sample = original_sample.reshape(1, -1).astype(np.float32)
            else:
                normalized_sample = normalize_ecg_signal(original_sample, method=norm_method)
                test_sample = normalized_sample.reshape(1, -1).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {'ecg_input': test_sample})
            raw_preds = outputs[0][0]
            pred_idx = np.argmax(raw_preds)
            
            # Update statistics
            class_total[true_label_str] += 1
            confusion_matrix[true_idx][pred_idx] += 1
            
            if pred_idx == true_idx:
                correct_predictions += 1
                class_correct[true_label_str] += 1
                
        except Exception as e:
            print(f"   Error with sample {i}: {e}")
            continue
    
    # Results
    overall_accuracy = correct_predictions / test_size * 100
    
    print(f"\n   📈 FINAL RESULTS:")
    print(f"   🎯 Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"   ✅ Correct: {correct_predictions}/{test_size}")
    
    # Per-class accuracy
    print(f"\n   📊 Per-Class Accuracy:")
    for label_str, total in class_total.items():
        if total > 0:
            correct = class_correct[label_str]
            accuracy = correct / total * 100
            class_name = class_names[label_to_idx[label_str]]
            print(f"     {class_name}: {accuracy:.1f}% ({correct}/{total})")
    
    # Assessment
    print(f"\n   🏆 MODEL ASSESSMENT:")
    if overall_accuracy >= 85:
        print("   🌟 EXCELLENT: Model working correctly!")
        print("   ✅ Ready for production use")
    elif overall_accuracy >= 70:
        print("   👍 GOOD: Model has reasonable performance")
        print("   ⚠️  May need some optimization")
    else:
        print("   ❌ POOR: Model needs significant improvement")
        print("   🔧 Check training data or model architecture")

if __name__ == "__main__":
    best_method = test_all_normalization_methods()
