"""
🎯 Re-validate Model with Corrected Mapping
Apply the corrected label mapping and re-evaluate model performance
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_corrected_mapping():
    """Load the corrected mapping from the investigation"""
    try:
        with open('corrected_mapping.json', 'r') as f:
            config = json.load(f)
        return config
    except:
        print("❌ No corrected mapping found. Run simple_mapping_fix.py first.")
        return None

def validate_with_corrected_mapping():
    print("🎯 RE-VALIDATING WITH CORRECTED MAPPING")
    print("=" * 60)
    
    # Load corrected mapping
    mapping_config = load_corrected_mapping()
    if not mapping_config:
        return
    
    corrected_mapping = mapping_config['best_mapping']
    reverse_mapping = mapping_config['reverse_mapping']
    
    print(f"✅ Using corrected mapping: {mapping_config['mapping_name']}")
    print(f"   Mapping: {corrected_mapping}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV1_1D(num_classes=5, width_multiplier=0.6)
    
    try:
        checkpoint = torch.load('../best_model_robust.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data
    X_test = np.load('../mod/combined_ecg_final/test/segments.npy')
    y_test = np.load('../mod/combined_ecg_final/test/labels.npy')
    
    print(f"✅ Test data: {X_test.shape}, {y_test.shape}")
    
    # Convert to tensor and get predictions
    X_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    
    # Apply corrected mapping
    y_corrected = np.array([corrected_mapping[label] for label in y_test])
    
    # Calculate metrics
    accuracy = accuracy_score(y_corrected, predictions)
    
    print(f"\n📊 CORRECTED VALIDATION RESULTS:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    class_names = [reverse_mapping[str(i)] for i in range(5)]
    report = classification_report(y_corrected, predictions, target_names=class_names, output_dict=True)
    
    print(f"\n📈 PER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        if class_name in report and isinstance(report[class_name], dict):
            metrics = report[class_name]
            if isinstance(metrics, dict) and all(key in metrics for key in ['precision', 'recall', 'f1-score', 'support']):
                print(f"  {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                      f"F1={metrics['f1-score']:.3f}, N={int(metrics['support'])}")
    
    # Overall metrics
    macro_f1 = 0.0
    weighted_f1 = 0.0
    
    if isinstance(report, dict):
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})
        
        if isinstance(macro_avg, dict):
            macro_f1 = macro_avg.get('f1-score', 0.0)
        
        if isinstance(weighted_avg, dict):
            weighted_f1 = weighted_avg.get('f1-score', 0.0)
    
    print(f"\n🎯 OVERALL METRICS:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Macro F1: {macro_f1:.4f}")
    print(f"   Weighted F1: {weighted_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_corrected, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Corrected Mapping\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('corrected_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confidence analysis
    correct_mask = (predictions == y_corrected)
    correct_confidences = np.max(probabilities[correct_mask], axis=1)
    incorrect_confidences = np.max(probabilities[~correct_mask], axis=1)
    
    print(f"\n🎲 CONFIDENCE ANALYSIS:")
    print(f"   Correct predictions: {correct_confidences.mean():.3f} ± {correct_confidences.std():.3f}")
    print(f"   Incorrect predictions: {incorrect_confidences.mean():.3f} ± {incorrect_confidences.std():.3f}")
    
    # Sample predictions
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    
    for idx in sample_indices:
        true_label_idx = y_corrected[idx]
        pred_label_idx = predictions[idx]
        true_label = reverse_mapping[str(true_label_idx)]
        pred_label = reverse_mapping[str(pred_label_idx)]
        confidence = probabilities[idx].max()
        
        status = "✅" if true_label_idx == pred_label_idx else "❌"
        print(f"   {status} True: {true_label} | Pred: {pred_label} | Conf: {confidence:.3f}")
    
    # Save results
    final_results = {
        'corrected_accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'mapping_used': corrected_mapping,
        'per_class_metrics': {}
    }
    
    # Add per-class metrics with proper type checking
    for i in range(5):
        class_name = class_names[i]
        if class_name in report and isinstance(report[class_name], dict):
            class_metrics = report[class_name]
            if isinstance(class_metrics, dict) and all(key in class_metrics for key in ['precision', 'recall', 'f1-score', 'support']):
                final_results['per_class_metrics'][class_name] = {
                    'precision': float(class_metrics['precision']),
                    'recall': float(class_metrics['recall']),
                    'f1': float(class_metrics['f1-score']),
                    'support': int(class_metrics['support'])
                }
    
    with open('final_validation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 RESULTS SAVED:")
    print("   - final_validation_results.json")
    print("   - corrected_confusion_matrix.png")
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    if accuracy > 0.9:
        print("   🚀 EXCELLENT! Model ready for deployment (>90% accuracy)")
        deployment_ready = "YES"
    elif accuracy > 0.8:
        print("   👍 GOOD! Model suitable for many applications (>80% accuracy)")
        deployment_ready = "CONDITIONAL"
    else:
        print("   ⚠️  MODEST: May need further optimization")
        deployment_ready = "NEEDS_WORK"
    
    print(f"   Deployment Ready: {deployment_ready}")
    print(f"   Corrected Accuracy: {accuracy*100:.2f}%")
    
    return accuracy, final_results

if __name__ == "__main__":
    validate_with_corrected_mapping()
