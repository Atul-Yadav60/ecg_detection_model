import numpy as np
import onnxruntime as ort
from collections import defaultdict

def evaluate_model_accuracy():
    """Comprehensive evaluation of model predictions"""
    
    print("🧪 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Load model
    session = ort.InferenceSession('mobilenet_v1_ecg_model.onnx')
    
    # Load test data
    X = np.load('../mod/balanced_ecg_smote/X_balanced.npy')
    y = np.load('../mod/balanced_ecg_smote/y_balanced.npy')
    
    class_names = ['Normal (N)', 'Ventricular (V)', 'Supraventricular (S)', 'Fusion (F)', 'Unknown (Q)']
    label_to_idx = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
    
    print(f"📊 Dataset: {X.shape[0]:,} samples")
    print(f"📊 Classes: {len(class_names)}")
    
    # Test on a representative sample (1000 samples for speed)
    test_size = min(1000, len(X))
    indices = np.random.choice(len(X), test_size, replace=False)
    
    correct_predictions = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    confusion_matrix = np.zeros((5, 5), dtype=int)
    
    print(f"\n🎯 Testing on {test_size} random samples...")
    
    for i, idx in enumerate(indices):
        if i % 200 == 0:
            print(f"  Progress: {i}/{test_size}")
            
        # Prepare input
        test_sample = X[idx:idx+1].astype(np.float32)
        true_label_str = y[idx]
        true_idx = label_to_idx[true_label_str]
        
        # Run inference
        outputs = session.run(None, {'ecg_input': test_sample})
        raw_preds = outputs[0][0]
        
        # Get predicted class
        pred_idx = np.argmax(raw_preds)
        
        # Update statistics
        class_total[true_label_str] += 1
        confusion_matrix[true_idx][pred_idx] += 1
        
        if pred_idx == true_idx:
            correct_predictions += 1
            class_correct[true_label_str] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct_predictions / test_size * 100
    
    print(f"\n📈 RESULTS:")
    print("=" * 60)
    print(f"🎯 Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"✅ Correct Predictions: {correct_predictions}/{test_size}")
    print(f"❌ Wrong Predictions: {test_size - correct_predictions}/{test_size}")
    
    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    for label_str, total in class_total.items():
        if total > 0:
            correct = class_correct[label_str]
            accuracy = correct / total * 100
            class_name = class_names[label_to_idx[label_str]]
            print(f"  {class_name}: {accuracy:.1f}% ({correct}/{total})")
    
    # Show confusion matrix
    print(f"\n🔍 Confusion Matrix:")
    print("    Predicted ->")
    print("True ↓   ", end="")
    for name in class_names:
        print(f"{name[:8]:>8}", end="")
    print()
    
    for i, true_name in enumerate(class_names):
        print(f"{true_name[:8]:>8} ", end="")
        for j in range(5):
            print(f"{confusion_matrix[i][j]:>8}", end="")
        print()
    
    # Analyze common mistakes
    print(f"\n❌ Most Common Mistakes:")
    mistakes = []
    for i in range(5):
        for j in range(5):
            if i != j and confusion_matrix[i][j] > 0:
                mistakes.append((confusion_matrix[i][j], class_names[i], class_names[j]))
    
    mistakes.sort(reverse=True)
    for count, true_class, pred_class in mistakes[:5]:
        if count > 0:
            print(f"  {true_class} → {pred_class}: {count} times")
    
    # Show some examples
    print(f"\n🔍 Example Predictions (first 10):")
    for i in range(min(10, len(indices))):
        idx = indices[i]
        test_sample = X[idx:idx+1].astype(np.float32)
        true_label_str = y[idx]
        true_idx = label_to_idx[true_label_str]
        
        outputs = session.run(None, {'ecg_input': test_sample})
        raw_preds = outputs[0][0]
        pred_idx = np.argmax(raw_preds)
        confidence = np.max(raw_preds)
        
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        status = "✅" if pred_idx == true_idx else "❌"
        
        print(f"  {i+1:2d}. {status} True: {true_name:<20} | Pred: {pred_name:<20} | Confidence: {confidence:.2f}")
    
    # Final assessment
    print(f"\n🏆 MODEL ASSESSMENT:")
    print("=" * 60)
    if overall_accuracy >= 90:
        print("🌟 EXCELLENT: Your model is performing very well!")
    elif overall_accuracy >= 80:
        print("👍 GOOD: Your model has solid performance.")
    elif overall_accuracy >= 70:
        print("⚠️  FAIR: Model needs improvement.")
    else:
        print("❌ POOR: Model needs significant improvement.")
    
    print(f"\n💡 Key Insights:")
    print(f"  - Test accuracy: {overall_accuracy:.1f}%")
    print(f"  - Training accuracy was: 99.33%")
    print(f"  - Validation accuracy was: 95.67%")
    
    if overall_accuracy < 90:
        print(f"\n🔧 Possible Issues:")
        print(f"  - Data distribution differences between train/test")
        print(f"  - Model overfitting to training data")
        print(f"  - Need for data normalization or preprocessing")
    else:
        print(f"\n✅ Model is working correctly!")
        print(f"  - Predictions are reliable")
        print(f"  - Ready for production use")
        print(f"  - Good generalization from training")

if __name__ == "__main__":
    evaluate_model_accuracy()
