#!/usr/bin/env python3
"""
Individual Model Report Generator
Generates detailed reports and graphs for each model separately
No comparison - just comprehensive analysis of each model
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
import time
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)

# Add src directory to path
sys.path.append('src')

def load_model_data(model_path, model_name):
    """Load model checkpoint and extract all data"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract all available data
        best_metrics = checkpoint.get('best_metrics', {})
        config = checkpoint.get('config', {})
        history = checkpoint.get('history', {})
        model_state = checkpoint.get('model_state_dict', {})
        
        # Calculate model specifications
        total_params = sum(p.numel() for p in model_state.values()) if model_state else 0
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        model_data = {
            'name': model_name,
            'path': model_path,
            'best_metrics': best_metrics,
            'config': config,
            'history': history,
            'model_size_mb': model_size_mb,
            'total_params': total_params,
            'timestamp': checkpoint.get('timestamp', 'Unknown'),
            'model_state_dict': model_state
        }
        
        print(f"SUCCESS {model_name} loaded successfully")
        print(f"   Epoch: {best_metrics.get('epoch', 'N/A')}")
        print(f"   Validation Accuracy: {best_metrics.get('val_acc', 0):.2f}%")
        print(f"   Model Size: {model_size_mb:.2f} MB")
        print(f"   Parameters: {total_params:,}")
        
        return model_data
        
    except Exception as e:
        print(f"ERROR loading {model_name}: {e}")
        return None

def evaluate_model_on_test(model_data):
    """Evaluate model on test set and get detailed metrics"""
    try:
        from train_mobilenet_optimized import OptimizedMobileNetTrainer
        
        print(f"Evaluating {model_data['name']} on test set...")
        
        # Initialize trainer
        trainer = OptimizedMobileNetTrainer('mobilenet', model_data['config'])
        trainer.load_balanced_data()
        trainer.setup_model()
        trainer.create_balanced_data_loaders()
        
        # Load model weights from the model_state_dict, not best_metrics
        trainer.model.load_state_dict(model_data['model_state_dict'])
        
        # Evaluate on test set
        trainer.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in trainer.test_loader:
                data = data.to(trainer.device)
                
                start_time = time.time()
                output = trainer.model(data)
                inference_time = (time.time() - start_time) * 1000 / data.size(0)
                inference_times.append(inference_time)
                
                pred = output.argmax(dim=1)
                probs = torch.softmax(output, dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate all metrics
        test_results = {
            'test_accuracy': accuracy_score(all_targets, all_preds) * 100,
            'test_f1_weighted': f1_score(all_targets, all_preds, average='weighted') * 100,
            'test_f1_macro': f1_score(all_targets, all_preds, average='macro') * 100,
            'test_precision': precision_score(all_targets, all_preds, average='weighted') * 100,
            'test_recall': recall_score(all_targets, all_preds, average='weighted') * 100,
            'test_balanced_accuracy': balanced_accuracy_score(all_targets, all_preds) * 100,
            'avg_inference_time': np.mean(inference_times),
            'confusion_matrix': confusion_matrix(all_targets, all_preds),
            'classification_report': classification_report(all_targets, all_preds, output_dict=True),
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
        
        print(f"Test evaluation complete:")
        print(f"   Test Accuracy: {test_results['test_accuracy']:.2f}%")
        print(f"   Test F1: {test_results['test_f1_weighted']:.2f}%")
        print(f"   Avg Inference: {test_results['avg_inference_time']:.2f}ms")
        
        return test_results
        
    except Exception as e:
        print(f"Error evaluating {model_data['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_individual_plots(model_data, test_results, output_dir):
    """Create comprehensive plots for individual model"""
    
    model_name = model_data['name'].replace(' ', '_').replace('(', '').replace(')', '')
    
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Training History (if available)
    if 'history' in model_data and model_data['history']:
        history = model_data['history']
        if 'epochs' in history:
            epochs = history['epochs']
            
            if 'train_acc' in history and 'val_acc' in history:
                axes[0,0].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
                axes[0,0].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
                axes[0,0].set_title('Training Progress - Accuracy', fontweight='bold')
                axes[0,0].set_xlabel('Epoch')
                axes[0,0].set_ylabel('Accuracy (%)')
                axes[0,0].legend()
                axes[0,0].grid(True, alpha=0.3)
            
            if 'train_loss' in history and 'val_loss' in history:
                ax_loss = axes[0,0].twinx()
                ax_loss.plot(epochs, history['train_loss'], 'g--', alpha=0.7, label='Train Loss')
                ax_loss.plot(epochs, history['val_loss'], 'm--', alpha=0.7, label='Val Loss')
                ax_loss.set_ylabel('Loss')
                ax_loss.legend(loc='upper right')
    else:
        axes[0,0].text(0.5, 0.5, 'Training History\nNot Available', 
                      ha='center', va='center', transform=axes[0,0].transAxes, fontsize=12)
        axes[0,0].set_title('Training Progress', fontweight='bold')
    
    # 2. Performance Metrics Bar Chart
    if test_results:
        metrics = ['Test Accuracy', 'Test F1', 'Precision', 'Recall', 'Balanced Acc']
        values = [
            test_results['test_accuracy'],
            test_results['test_f1_weighted'], 
            test_results['test_precision'],
            test_results['test_recall'],
            test_results['test_balanced_accuracy']
        ]
        
        bars = axes[0,1].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'], alpha=0.8)
        axes[0,1].set_title('Test Set Performance', fontweight='bold')
        axes[0,1].set_ylabel('Score (%)')
        axes[0,1].set_ylim(0, 100)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0,1].annotate(f'{height:.1f}%',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=10)
        
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. Gap Analysis
    best_metrics = model_data['best_metrics']
    gap_metrics = ['Generalization Gap', 'Accuracy Gap', 'F1 Gap']
    gap_values = [
        best_metrics.get('generalization_gap', 0),
        best_metrics.get('accuracy_gap', 0),
        best_metrics.get('f1_gap', 0)
    ]
    gap_targets = [5, 3.5, 2.5]  # Target centers
    gap_colors = ['red' if gap_values[i] > gap_targets[i] else 'green' for i in range(3)]
    
    bars = axes[0,2].bar(gap_metrics, gap_values, color=gap_colors, alpha=0.7)
    
    # Add target lines
    axes[0,2].axhline(y=5, color='red', linestyle='--', alpha=0.8, label='Gen Limit (5%)')
    axes[0,2].axhline(y=2, color='blue', linestyle='--', alpha=0.8, label='Gap Min (2%)')
    axes[0,2].axhline(y=4, color='orange', linestyle='--', alpha=0.8, label='F1 Max (4%)')
    
    axes[0,2].set_title('Gap Analysis', fontweight='bold')
    axes[0,2].set_ylabel('Gap (%)')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, gap_values):
        axes[0,2].annotate(f'{value:.2f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, value),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=10)
    
    # 4. Model Specifications
    spec_labels = ['Size (MB)', 'Inference (ms)', 'Parameters (M)']
    spec_values = [
        model_data['model_size_mb'],
        best_metrics.get('inference_time', 0),
        model_data['total_params'] / 1e6
    ]
    
    bars = axes[1,0].bar(spec_labels, spec_values, color=['purple', 'orange', 'brown'], alpha=0.8)
    
    # Add target lines
    axes[1,0].axhline(y=5, color='red', linestyle='--', alpha=0.8, label='Mobile Limit (5MB)')
    axes[1,0].axhline(y=50, color='orange', linestyle='--', alpha=0.8, label='Real-time Limit (50ms)')
    
    axes[1,0].set_title('Model Specifications', fontweight='bold')
    axes[1,0].set_ylabel('Value')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Confusion Matrix
    if test_results and 'confusion_matrix' in test_results:
        cm = test_results['confusion_matrix']
        im = axes[1,1].imshow(cm, cmap='Blues', alpha=0.8)
        axes[1,1].set_title('Confusion Matrix', fontweight='bold')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1,1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
        
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
        
        # Add class labels
        class_names = ['F', 'N', 'Q', 'S', 'V']
        axes[1,1].set_xticks(range(len(class_names)))
        axes[1,1].set_yticks(range(len(class_names)))
        axes[1,1].set_xticklabels(class_names)
        axes[1,1].set_yticklabels(class_names)
    
    # 6. Per-Class Performance
    if test_results and 'classification_report' in test_results:
        class_report = test_results['classification_report']
        class_names = ['F', 'N', 'Q', 'S', 'V']
        
        precisions = [class_report.get(str(i), {}).get('precision', 0) * 100 for i in range(5)]
        recalls = [class_report.get(str(i), {}).get('recall', 0) * 100 for i in range(5)]
        f1s = [class_report.get(str(i), {}).get('f1-score', 0) * 100 for i in range(5)]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        axes[1,2].bar(x - width, precisions, width, label='Precision', color='skyblue', alpha=0.8)
        axes[1,2].bar(x, recalls, width, label='Recall', color='lightcoral', alpha=0.8)
        axes[1,2].bar(x + width, f1s, width, label='F1-Score', color='lightgreen', alpha=0.8)
        
        axes[1,2].set_title('Per-Class Performance', fontweight='bold')
        axes[1,2].set_xlabel('ECG Classes')
        axes[1,2].set_ylabel('Score (%)')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(class_names)
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_data["name"]} - Comprehensive Analysis Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'{output_dir}/{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def generate_individual_report(model_data, test_results, output_dir):
    """Generate detailed markdown report for individual model"""
    
    best_metrics = model_data['best_metrics']
    config = model_data['config']
    
    report = f"""# {model_data['name']} - Detailed Analysis Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Model Path**: {model_data['path']}

## Model Overview

### Basic Information
- **Model Name**: {model_data['name']}
- **Training Completed**: Epoch {best_metrics.get('epoch', 'N/A')}
- **Model Size**: {model_data['model_size_mb']:.2f} MB
- **Total Parameters**: {model_data['total_params']:,}
- **Timestamp**: {model_data['timestamp']}

### Architecture Configuration
- **Width Multiplier**: {config.get('width_multiplier', 'N/A')}
- **Dropout Rate**: {config.get('dropout_rate', 'N/A')}
- **Batch Size**: {config.get('batch_size', 'N/A')}
- **Learning Rate**: {config.get('learning_rate', 'N/A')}
- **Weight Decay**: {config.get('weight_decay', 'N/A')}

## Performance Metrics

### Validation Performance
- **Validation Accuracy**: {best_metrics.get('val_acc', 0):.2f}%
- **Validation F1**: {best_metrics.get('val_f1', 0):.2f}%
- **Balanced Accuracy**: {best_metrics.get('val_balanced_acc', 0):.2f}%

"""
    
    if test_results:
        report += f"""### Test Set Performance
- **Test Accuracy**: {test_results['test_accuracy']:.2f}%
- **Test F1 (Weighted)**: {test_results['test_f1_weighted']:.2f}%
- **Test F1 (Macro)**: {test_results['test_f1_macro']:.2f}%
- **Test Precision**: {test_results['test_precision']:.2f}%
- **Test Recall**: {test_results['test_recall']:.2f}%
- **Balanced Accuracy**: {test_results['test_balanced_accuracy']:.2f}%

"""
    
    report += f"""## Generalization Analysis

### Gap Metrics
- **Generalization Gap**: {best_metrics.get('generalization_gap', 0):.2f}%
- **Accuracy Gap**: {best_metrics.get('accuracy_gap', 0):.2f}%
- **F1 Gap**: {best_metrics.get('f1_gap', 0):.2f}%

### Gap Compliance Check
- **Generalization Gap < 5%**: {"PASS" if best_metrics.get('generalization_gap', 0) < 5 else "FAIL"} ({best_metrics.get('generalization_gap', 0):.2f}%)
- **Accuracy Gap 2-5%**: {"PASS" if 2 <= best_metrics.get('accuracy_gap', 0) <= 5 else "FAIL"} ({best_metrics.get('accuracy_gap', 0):.2f}%)
- **F1 Gap 1-4%**: {"PASS" if 1 <= best_metrics.get('f1_gap', 0) <= 4 else "FAIL"} ({best_metrics.get('f1_gap', 0):.2f}%)

## Performance Specifications

### Speed & Size
- **Average Inference Time**: {best_metrics.get('inference_time', 0):.2f} ms/sample
- **Model Size**: {model_data['model_size_mb']:.2f} MB
- **Parameter Count**: {model_data['total_params']:,}

### Deployment Readiness
- **Real-time Ready (<50ms)**: {"YES" if best_metrics.get('inference_time', 0) < 50 else "NO"} ({best_metrics.get('inference_time', 0):.2f}ms)
- **Mobile Ready (<5MB)**: {"YES" if model_data['model_size_mb'] < 5 else "NO"} ({model_data['model_size_mb']:.2f}MB)
- **Clinical Grade (>90% acc)**: {"YES" if best_metrics.get('val_acc', 0) > 90 else "NO"} ({best_metrics.get('val_acc', 0):.2f}%)

"""
    
    if test_results and 'classification_report' in test_results:
        report += """## Per-Class Performance Analysis

| Class | Precision (%) | Recall (%) | F1-Score (%) | Support |
|-------|---------------|------------|--------------|---------|
"""
        
        class_names = {'0': 'F (Fusion)', '1': 'N (Normal)', '2': 'Q (Unknown)', 
                      '3': 'S (Supraventricular)', '4': 'V (Ventricular)'}
        
        class_report = test_results['classification_report']
        for class_id in ['0', '1', '2', '3', '4']:
            if class_id in class_report:
                metrics = class_report[class_id]
                class_name = class_names[class_id]
                precision = metrics['precision'] * 100
                recall = metrics['recall'] * 100
                f1 = metrics['f1-score'] * 100
                support = int(metrics['support'])
                
                report += f"| {class_name} | {precision:.1f} | {recall:.1f} | {f1:.1f} | {support} |\n"
    
    # Training Configuration Details
    report += f"""

## Training Configuration Details

### Core Training Parameters
- **Epochs Trained**: {best_metrics.get('epoch', 'N/A')}
- **Batch Size**: {config.get('batch_size', 'N/A')}
- **Learning Rate**: {config.get('learning_rate', 'N/A')}
- **Weight Decay**: {config.get('weight_decay', 'N/A')}

### Regularization Settings
- **Dropout Rate**: {config.get('dropout_rate', 'N/A')}
- **Label Smoothing**: {config.get('label_smoothing', 'N/A')}
- **Gradient Clip Norm**: {config.get('gradient_clip_norm', 'N/A')}

### Architecture Settings
- **Width Multiplier**: {config.get('width_multiplier', 'N/A')}
- **Use Mixed Precision**: {config.get('use_mixed_precision', 'N/A')}

### Training Strategy
- **Early Stopping Metric**: {config.get('early_stopping_metric', 'N/A')}
- **Early Stopping Patience**: {config.get('early_stopping_patience', 'N/A')}
- **Scheduler Factor**: {config.get('factor', 'N/A')}

## Deployment Assessment

### Mobile Deployment Readiness
{"READY FOR MOBILE" if model_data['model_size_mb'] < 5 and best_metrics.get('inference_time', 0) < 50 else "NEEDS OPTIMIZATION"}

- Size Requirement: {"Met" if model_data['model_size_mb'] < 5 else "Too Large"} ({model_data['model_size_mb']:.2f}MB < 5MB)
- Speed Requirement: {"Met" if best_metrics.get('inference_time', 0) < 50 else "Too Slow"} ({best_metrics.get('inference_time', 0):.2f}ms < 50ms)
- Accuracy Requirement: {"Met" if best_metrics.get('val_acc', 0) > 90 else "Too Low"} ({best_metrics.get('val_acc', 0):.2f}% > 90%)

### Clinical Deployment Readiness
{"READY FOR CLINICAL USE" if best_metrics.get('val_acc', 0) > 93 and best_metrics.get('generalization_gap', 0) < 5 else "NEEDS VALIDATION"}

- Clinical Accuracy: {"Excellent" if best_metrics.get('val_acc', 0) > 95 else "Good" if best_metrics.get('val_acc', 0) > 90 else "Insufficient"} ({best_metrics.get('val_acc', 0):.2f}%)
- Generalization: {"Excellent" if best_metrics.get('generalization_gap', 0) < 5 else "Poor"} ({best_metrics.get('generalization_gap', 0):.2f}% gap)
- Class Balance: {"Good" if test_results and test_results['test_balanced_accuracy'] > 85 else "Check"} ({test_results['test_balanced_accuracy']:.1f}% bal acc)""" if test_results else "N/A"
    
    report += f"""
---
**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    return report

def main():
    """Generate individual reports for all 3 models"""
    print("INDIVIDUAL MODEL REPORT GENERATOR")
    print("="*50)
    
    # Define all 3 models
    models_to_analyze = [
        ("outputs/models/approach1_performance_first/mobilenet_best.pth", "Approach_1_Performance_First"),
        ("outputs/models/strategy2_tight_gaps/mobilenet_best.pth", "Approach_2_Current_Best"),
        ("outputs/models/strategy2_tight_gaps/mobilenet_epoch38_backup.pth", "Approach_2_Backup_Model")
    ]
    
    # Create output directory
    output_dir = 'outputs/individual_reports'
    os.makedirs(output_dir, exist_ok=True)
    
    successful_reports = 0
    
    # Generate report for each model
    for model_path, model_name in models_to_analyze:
        print(f"\nPROCESSING: {model_name}")
        print("-" * 40)
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model not found: {model_path}")
            continue
        
        # Load model data
        model_data = load_model_data(model_path, model_name)
        if not model_data:
            continue
        
        # Evaluate on test set
        test_results = evaluate_model_on_test(model_data)
        
        # Create plots
        plot_file = create_individual_plots(model_data, test_results, output_dir)
        
        # Generate report
        report_content = generate_individual_report(model_data, test_results, output_dir)
        
        # Save report
        report_filename = f'{output_dir}/{model_name}_report.md'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save raw data
        raw_data = {
            'model_info': {
                'name': model_data['name'],
                'epoch': model_data['best_metrics'].get('epoch', 0),
                'size_mb': model_data['model_size_mb'],
                'parameters': model_data['total_params']
            },
            'performance': {
                'val_accuracy': model_data['best_metrics'].get('val_acc', 0),
                'val_f1': model_data['best_metrics'].get('val_f1', 0),
                'test_accuracy': test_results['test_accuracy'] if test_results else 0,
                'test_f1': test_results['test_f1_weighted'] if test_results else 0
            },
            'gaps': {
                'generalization_gap': model_data['best_metrics'].get('generalization_gap', 0),
                'accuracy_gap': model_data['best_metrics'].get('accuracy_gap', 0),
                'f1_gap': model_data['best_metrics'].get('f1_gap', 0)
            },
            'deployment': {
                'inference_time_ms': model_data['best_metrics'].get('inference_time', 0),
                'mobile_ready': model_data['model_size_mb'] < 5,
                'real_time_ready': model_data['best_metrics'].get('inference_time', 0) < 50
            }
        }
        
        with open(f'{output_dir}/{model_name}_data.json', 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        successful_reports += 1
        print(f"SUCCESS: {model_name} report generated successfully")
    
    # Generate summary
    print(f"\nINDIVIDUAL REPORTS GENERATED!")
    print(f"Location: {output_dir}/")
    print(f"Reports Generated: {successful_reports}")
    print(f"\nFiles created:")
    for model_path, model_name in models_to_analyze:
        if os.path.exists(model_path):
            print(f"   {model_name}_report.md")
            print(f"   {model_name}_analysis.png")
            print(f"   {model_name}_data.json")

if __name__ == "__main__":
    main()