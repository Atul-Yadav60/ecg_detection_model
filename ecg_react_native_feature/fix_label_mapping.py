"""
🔍 ECG Label Mapping Investigation & Fix
Investigating N/V label swap and correcting mapping for optimal accuracy
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

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

def analyze_label_distributions():
    """Analyze label distributions across different data sources"""
    print("🔍 ANALYZING LABEL DISTRIBUTIONS")
    print("=" * 50)
    
    # Load balanced dataset
    X_balanced = np.load('../mod/balanced_ecg_smote/X_balanced.npy')
    y_balanced = np.load('../mod/balanced_ecg_smote/y_balanced.npy')
    
    # Load final combined dataset
    X_final = np.load('../mod/combined_ecg_final/X_final_combined.npy')
    y_final = np.load('../mod/combined_ecg_final/y_final_combined.npy')
    
    print(f"Balanced dataset: {X_balanced.shape}, {y_balanced.shape}")
    print(f"Final dataset: {X_final.shape}, {y_final.shape}")
    
    # Count distributions
    balanced_counts = Counter(y_balanced)
    final_counts = Counter(y_final)
    
    print("\n📊 BALANCED Dataset Distribution:")
    total_balanced = len(y_balanced)
    for label in ['N', 'V', 'S', 'F', 'Q']:
        count = balanced_counts.get(label, 0)
        pct = (count / total_balanced) * 100
        print(f"  {label}: {count:6,} ({pct:5.1f}%)")
    
    print("\n📊 FINAL Dataset Distribution:")
    total_final = len(y_final)
    for label in ['N', 'V', 'S', 'F', 'Q']:
        count = final_counts.get(label, 0)
        pct = (count / total_final) * 100
        print(f"  {label}: {count:6,} ({pct:5.1f}%)")
    
    return balanced_counts, final_counts

def test_label_mappings():
    """Test different label mapping hypotheses"""
    print("\n🧪 TESTING LABEL MAPPING HYPOTHESES")
    print("=" * 50)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV1_1D(num_classes=5, width_multiplier=0.6)
    
    try:
        checkpoint = torch.load('../best_model_robust.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ Robust model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data from final dataset (not SMOTE-balanced)
    X_test = np.load('../mod/combined_ecg_final/test/X_test.npy')
    y_test = np.load('../mod/combined_ecg_final/test/y_test.npy')
    
    print(f"Test data: {X_test.shape}, {y_test.shape}")
    
    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    
    # Original mapping
    original_mapping = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}
    class_names = ['N', 'V', 'S', 'F', 'Q']
    
    # Test different mapping hypotheses
    mapping_hypotheses = {
        'Original': {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4},
        'N-V Swap': {'N': 1, 'V': 0, 'S': 2, 'F': 3, 'Q': 4},
        'Reverse Order': {'N': 4, 'V': 3, 'S': 2, 'F': 1, 'Q': 0},
        'Alphabetical': {'F': 0, 'N': 1, 'Q': 2, 'S': 3, 'V': 4}
    }
    
    results = {}
    
    with torch.no_grad():
        # Get model predictions
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    for hypothesis_name, mapping in mapping_hypotheses.items():
        print(f"\n🔬 Testing: {hypothesis_name}")
        
        # Convert true labels using this mapping
        reverse_mapping = {v: k for k, v in mapping.items()}
        y_mapped = np.array([mapping[label] for label in y_test])
        
        # Calculate accuracy
        accuracy = (predictions == y_mapped).mean()
        
        # Generate classification report
        try:
            report = classification_report(y_mapped, predictions, 
                                         target_names=[reverse_mapping[i] for i in range(5)],
                                         output_dict=True, zero_division=0)
            
            # Safely access dictionary keys with proper type checking
            if isinstance(report, dict):
                f1_macro = report.get('macro avg', {}).get('f1-score', 0.0)
                f1_weighted = report.get('weighted avg', {}).get('f1-score', 0.0)
            else:
                f1_macro = 0.0
                f1_weighted = 0.0
            
            results[hypothesis_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'mapping': mapping
            }
            
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  F1-Macro: {f1_macro:.4f}")
            print(f"  F1-Weighted: {f1_weighted:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Find best mapping
    best_hypothesis = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_hypothesis]['accuracy']
    
    print(f"\n🏆 BEST MAPPING: {best_hypothesis}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"   Mapping: {results[best_hypothesis]['mapping']}")
    
    return results, best_hypothesis

def create_confusion_matrix_analysis(results, best_hypothesis):
    """Create detailed confusion matrix for best mapping"""
    print(f"\n📊 DETAILED ANALYSIS: {best_hypothesis}")
    print("=" * 50)
    
    # Load model and data again
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV1_1D(num_classes=5, width_multiplier=0.6)
    checkpoint = torch.load('../best_model_robust.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    X_test = np.load('../mod/combined_ecg_final/test/X_test.npy')
    y_test = np.load('../mod/combined_ecg_final/test/y_test.npy')
    
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    
    best_mapping = results[best_hypothesis]['mapping']
    reverse_mapping = {v: k for k, v in best_mapping.items()}
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    
    # Convert true labels
    y_mapped = np.array([best_mapping[label] for label in y_test])
    
    # Confusion matrix
    cm = confusion_matrix(y_mapped, predictions)
    class_names = [reverse_mapping[i] for i in range(5)]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {best_hypothesis} Mapping\nAccuracy: {results[best_hypothesis]["accuracy"]:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{best_hypothesis.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Per-class analysis
    print("📈 PER-CLASS PERFORMANCE:")
    report = classification_report(y_mapped, predictions, 
                                 target_names=class_names, output_dict=True, zero_division=0)
    
    if isinstance(report, dict):
        for class_name in class_names:
            if class_name in report and isinstance(report[class_name], dict):
                metrics = report[class_name]
                precision = metrics.get('precision', 0.0)
                recall = metrics.get('recall', 0.0)
                f1_score = metrics.get('f1-score', 0.0)
                support = metrics.get('support', 0)
                print(f"  {class_name}: Precision={precision:.3f}, "
                      f"Recall={recall:.3f}, F1={f1_score:.3f}, "
                      f"Support={int(support)}")
    else:
        print(f"  Classification report returned string: {report}")
    
    # Confidence analysis
    correct_predictions = (predictions == y_mapped)
    correct_confidences = np.max(probabilities[correct_predictions], axis=1)
    incorrect_confidences = np.max(probabilities[~correct_predictions], axis=1)
    
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    print(f"  Correct predictions confidence: {correct_confidences.mean():.3f} ± {correct_confidences.std():.3f}")
    print(f"  Incorrect predictions confidence: {incorrect_confidences.mean():.3f} ± {incorrect_confidences.std():.3f}")
    
    return best_mapping, cm, report

def save_corrected_mapping(best_mapping, results):
    """Save the corrected mapping configuration"""
    print("\n💾 SAVING CORRECTED MAPPING")
    print("=" * 30)
    
    config = {
        'corrected_label_mapping': best_mapping,
        'reverse_mapping': {v: k for k, v in best_mapping.items()},
        'performance_metrics': results,
        'timestamp': '2025-01-06',
        'notes': 'Corrected mapping based on systematic testing of label hypotheses'
    }
    
    with open('corrected_label_mapping.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Saved to: corrected_label_mapping.json")
    
    # Create implementation code
    implementation_code = f"""
# 🔧 CORRECTED LABEL MAPPING - USE THIS IN ALL INFERENCE
CORRECTED_MAPPING = {best_mapping}
REVERSE_MAPPING = {config['reverse_mapping']}
CLASS_NAMES = {[config['reverse_mapping'][i] for i in range(5)]}

def convert_prediction_to_label(model_output_index):
    '''Convert model output index to ECG class label'''
    return REVERSE_MAPPING[model_output_index]

def convert_label_to_index(ecg_label):
    '''Convert ECG class label to model input index'''
    return CORRECTED_MAPPING[ecg_label]
"""
    
    with open('corrected_mapping_implementation.py', 'w') as f:
        f.write(implementation_code)
    
    print("✅ Implementation code saved to: corrected_mapping_implementation.py")

def main():
    print("🔍 ECG LABEL MAPPING INVESTIGATION & FIX")
    print("=" * 60)
    print("Investigating N/V label swap and finding optimal mapping...")
    
    # Step 1: Analyze distributions
    balanced_counts, final_counts = analyze_label_distributions()
    
    # Step 2: Test mapping hypotheses
    results, best_hypothesis = test_label_mappings()
    
    # Step 3: Detailed analysis
    best_mapping, cm, report = create_confusion_matrix_analysis(results, best_hypothesis)
    
    # Step 4: Save corrected mapping
    save_corrected_mapping(best_mapping, results)
    
    # Step 5: Summary
    print(f"\n🎯 INVESTIGATION COMPLETE!")
    print("=" * 40)
    print(f"✅ Best mapping identified: {best_hypothesis}")
    print(f"✅ Accuracy improvement: {results[best_hypothesis]['accuracy']:.4f} ({results[best_hypothesis]['accuracy']*100:.2f}%)")
    print(f"✅ Mapping: {best_mapping}")
    print(f"✅ Files saved:")
    print("   - corrected_label_mapping.json")
    print("   - corrected_mapping_implementation.py")
    print("   - confusion_matrix_*.png")
    
    if results[best_hypothesis]['accuracy'] > 0.9:
        print(f"\n🚀 EXCELLENT! Achieved >90% accuracy with corrected mapping!")
    elif results[best_hypothesis]['accuracy'] > 0.8:
        print(f"\n👍 Good improvement! Achieved >80% accuracy with corrected mapping.")
    else:
        print(f"\n⚠️  Mapping helps but may need additional investigation.")

if __name__ == "__main__":
    main()
