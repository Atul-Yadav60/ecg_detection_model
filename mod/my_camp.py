#!/usr/bin/env python3
"""
Comprehensive Model Comparison Report Generator
Compares Approach 1 (Performance-First) vs Approach 2 (Generalization-First)
Generates detailed analysis, plots, and deployment recommendations
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score
)

# Add src directory to path
sys.path.append('src')
from train_mobilenet_optimized import OptimizedMobileNetTrainer

class ModelComparisonReporter:
    """
    Comprehensive comparison between trained models
    Analyzes performance, generalization, deployment readiness
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.trainer = None
        
    def load_model_info(self, model_path, model_name):
        """Load model checkpoint and extract comprehensive information"""
        print(f"\n📊 Loading {model_name} from {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None
            
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Extract model information
            model_info = {
                'name': model_name,
                'path': model_path,
                'epoch': checkpoint.get('best_metrics', {}).get('epoch', 'Unknown'),
                'val_acc': checkpoint.get('best_metrics', {}).get('val_acc', 0),
                'val_f1': checkpoint.get('best_metrics', {}).get('val_f1', 0),
                'val_balanced_acc': checkpoint.get('best_metrics', {}).get('val_balanced_acc', 0),
                'generalization_gap': checkpoint.get('best_metrics', {}).get('generalization_gap', 0),
                'accuracy_gap': checkpoint.get('best_metrics', {}).get('accuracy_gap', 0),
                'f1_gap': checkpoint.get('best_metrics', {}).get('f1_gap', 0),
                'inference_time': checkpoint.get('best_metrics', {}).get('inference_time', 0),
                'config': checkpoint.get('config', {}),
                'history': checkpoint.get('history', {}),
                'timestamp': checkpoint.get('timestamp', 'Unknown'),
                'model_state': checkpoint.get('model_state_dict'),
                'total_params': 0,  # Will calculate later
                'model_size_mb': 0  # Will calculate later
            }
            
            # Calculate model size
            if model_info['model_state']:
                total_params = sum(p.numel() for p in model_info['model_state'].values())
                model_info['total_params'] = total_params
                model_info['model_size_mb'] = total_params * 4 / (1024 * 1024)
            
            self.models[model_name] = model_info
            
            print(f"✅ {model_name} loaded successfully:")
            print(f"   Epoch: {model_info['epoch']}")
            print(f"   Val Accuracy: {model_info['val_acc']:.2f}%")
            print(f"   Val F1: {model_info['val_f1']:.2f}%")
            print(f"   Gen Gap: {model_info['generalization_gap']:.2f}%")
            print(f"   Model Size: {model_info['model_size_mb']:.2f} MB")
            print(f"   Inference: {model_info['inference_time']:.2f}ms")
            
            return model_info
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return None
    
    def evaluate_model_on_test_set(self, model_info):
        """Evaluate model on test set and return comprehensive metrics"""
        print(f"\n🔬 Evaluating {model_info['name']} on test set...")
        
        try:
            # Initialize trainer if not done
            if self.trainer is None:
                self.trainer = OptimizedMobileNetTrainer('mobilenet', model_info['config'])
                self.trainer.load_balanced_data()
                self.trainer.setup_model()
                self.trainer.create_balanced_data_loaders()
            
            # Evaluate on test set
            test_results = self.trainer.evaluate_test_set(model_info['path'])
            
            # Add detailed per-class analysis
            self.trainer.model.eval()
            all_preds = []
            all_targets = []
            all_probs = []
            inference_times = []
            
            with torch.no_grad():
                for data, target in self.trainer.test_loader:
                    data = data.to(self.trainer.device)
                    
                    start_time = time.time()
                    output = self.trainer.model(data)
                    inference_time = (time.time() - start_time) * 1000 / data.size(0)
                    inference_times.append(inference_time)
                    
                    pred = output.argmax(dim=1)
                    probs = torch.softmax(output, dim=1)
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            # Calculate comprehensive metrics
            detailed_results = {
                'test_accuracy': accuracy_score(all_targets, all_preds) * 100,
                'test_f1_weighted': f1_score(all_targets, all_preds, average='weighted') * 100,
                'test_f1_macro': f1_score(all_targets, all_preds, average='macro') * 100,
                'test_balanced_accuracy': balanced_accuracy_score(all_targets, all_preds) * 100,
                'test_precision': precision_score(all_targets, all_preds, average='weighted') * 100,
                'test_recall': recall_score(all_targets, all_preds, average='weighted') * 100,
                'avg_inference_time': np.mean(inference_times),
                'std_inference_time': np.std(inference_times),
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times),
                'all_predictions': all_preds,
                'all_targets': all_targets,
                'all_probabilities': all_probs
            }
            
            # Calculate per-class metrics
            report_dict = classification_report(all_targets, all_preds, output_dict=True)
            detailed_results['per_class_metrics'] = report_dict
            
            # Calculate confusion matrix
            detailed_results['confusion_matrix'] = confusion_matrix(all_targets, all_preds)
            
            print(f"✅ {model_info['name']} test evaluation complete:")
            print(f"   Test Accuracy: {detailed_results['test_accuracy']:.2f}%")
            print(f"   Test F1 (Weighted): {detailed_results['test_f1_weighted']:.2f}%")
            print(f"   Test F1 (Macro): {detailed_results['test_f1_macro']:.2f}%")
            print(f"   Avg Inference: {detailed_results['avg_inference_time']:.2f}ms")
            
            return detailed_results
            
        except Exception as e:
            print(f"❌ Error evaluating {model_info['name']}: {e}")
            return None
    
    def generate_comparison_report(self, approach1_path, approach2_path, output_dir='outputs/comparison'):
        """Generate comprehensive comparison report"""
        print(f"\n🏆 GENERATING COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load both models
        approach1_info = self.load_model_info(approach1_path, "Approach 1 (Performance-First)")
        approach2_info = self.load_model_info(approach2_path, "Approach 2 (Generalization-First)")
        
        if not approach1_info or not approach2_info:
            print("❌ Cannot generate comparison - one or both models missing")
            return
        
        # Evaluate both models on test set
        approach1_results = self.evaluate_model_on_test_set(approach1_info)
        approach2_results = self.evaluate_model_on_test_set(approach2_info)
        
        if not approach1_results or not approach2_results:
            print("❌ Cannot generate comparison - evaluation failed")
            return
        
        # Generate comprehensive comparison
        self.create_comparison_tables(approach1_info, approach2_info, 
                                    approach1_results, approach2_results, output_dir)
        
        self.create_comparison_plots(approach1_info, approach2_info, 
                                   approach1_results, approach2_results, output_dir)
        
        self.create_deployment_comparison(approach1_info, approach2_info, 
                                        approach1_results, approach2_results, output_dir)
        
        self.create_final_report(approach1_info, approach2_info, 
                               approach1_results, approach2_results, output_dir)
        
        print(f"\n🎉 COMPREHENSIVE COMPARISON REPORT GENERATED!")
        print(f"📁 Location: {output_dir}/")
        print(f"📊 Files created:")
        print(f"   - comparison_report.md (Main report)")
        print(f"   - comparison_plots.png (Visual analysis)")
        print(f"   - deployment_comparison.json (Deployment specs)")
        print(f"   - performance_comparison.csv (Raw metrics)")
    
    def create_comparison_tables(self, app1_info, app2_info, app1_results, app2_results, output_dir):
        """Create detailed comparison tables"""
        
        # Training Configuration Comparison
        config_comparison = {
            'Parameter': [
                'Training Approach', 'Final Epoch', 'Batch Size', 'Learning Rate', 
                'Weight Decay', 'Dropout Rate', 'Width Multiplier', 'Early Stopping Patience',
                'Label Smoothing', 'Gradient Clip Norm', 'Training Time (Est.)'
            ],
            'Approach 1 (Performance-First)': [
                'Max accuracy focus',
                app1_info['epoch'],
                app1_info['config'].get('batch_size', 'N/A'),
                app1_info['config'].get('learning_rate', 'N/A'),
                app1_info['config'].get('weight_decay', 'N/A'),
                app1_info['config'].get('dropout_rate', 'N/A'),
                app1_info['config'].get('width_multiplier', 'N/A'),
                app1_info['config'].get('early_stopping_patience', 'N/A'),
                app1_info['config'].get('label_smoothing', 'N/A'),
                app1_info['config'].get('gradient_clip_norm', 'N/A'),
                f"~{int(app1_info['epoch']) * 10} min"
            ],
            'Approach 2 (Generalization-First)': [
                'Tight gaps focus',
                app2_info['epoch'],
                app2_info['config'].get('batch_size', 'N/A'),
                app2_info['config'].get('learning_rate', 'N/A'),
                app2_info['config'].get('weight_decay', 'N/A'),
                app2_info['config'].get('dropout_rate', 'N/A'),
                app2_info['config'].get('width_multiplier', 'N/A'),
                app2_info['config'].get('early_stopping_patience', 'N/A'),
                app2_info['config'].get('label_smoothing', 'N/A'),
                app2_info['config'].get('gradient_clip_norm', 'N/A'),
                f"~{int(app2_info['epoch']) * 10} min"
            ]
        }
        
        # Performance Comparison
        performance_comparison = {
            'Metric': [
                'Test Accuracy (%)', 'Test F1 Weighted (%)', 'Test F1 Macro (%)',
                'Balanced Accuracy (%)', 'Test Precision (%)', 'Test Recall (%)',
                'Validation Accuracy (%)', 'Validation F1 (%)', 'Generalization Gap (%)',
                'Accuracy Gap (%)', 'F1 Gap (%)', 'Avg Inference (ms)', 'Model Size (MB)',
                'Total Parameters', 'Real-time Ready', 'Mobile Ready (<5MB)'
            ],
            'Approach 1': [
                f"{app1_results['test_accuracy']:.2f}",
                f"{app1_results['test_f1_weighted']:.2f}",
                f"{app1_results['test_f1_macro']:.2f}",
                f"{app1_results['test_balanced_accuracy']:.2f}",
                f"{app1_results['test_precision']:.2f}",
                f"{app1_results['test_recall']:.2f}",
                f"{app1_info['val_acc']:.2f}",
                f"{app1_info['val_f1']:.2f}",
                f"{app1_info['generalization_gap']:.2f}",
                f"{app1_info['accuracy_gap']:.2f}",
                f"{app1_info['f1_gap']:.2f}",
                f"{app1_results['avg_inference_time']:.2f}",
                f"{app1_info['model_size_mb']:.2f}",
                f"{app1_info['total_params']:,}",
                "✅ YES" if app1_results['avg_inference_time'] < 50 else "❌ NO",
                "✅ YES" if app1_info['model_size_mb'] < 5 else "❌ NO"
            ],
            'Approach 2': [
                f"{app2_results['test_accuracy']:.2f}",
                f"{app2_results['test_f1_weighted']:.2f}",
                f"{app2_results['test_f1_macro']:.2f}",
                f"{app2_results['test_balanced_accuracy']:.2f}",
                f"{app2_results['test_precision']:.2f}",
                f"{app2_results['test_recall']:.2f}",
                f"{app2_info['val_acc']:.2f}",
                f"{app2_info['val_f1']:.2f}",
                f"{app2_info['generalization_gap']:.2f}",
                f"{app2_info['accuracy_gap']:.2f}",
                f"{app2_info['f1_gap']:.2f}",
                f"{app2_results['avg_inference_time']:.2f}",
                f"{app2_info['model_size_mb']:.2f}",
                f"{app2_info['total_params']:,}",
                "✅ YES" if app2_results['avg_inference_time'] < 50 else "❌ NO",
                "✅ YES" if app2_info['model_size_mb'] < 5 else "❌ NO"
            ]
        }
        
        # Save as CSV
        pd.DataFrame(config_comparison).to_csv(f'{output_dir}/config_comparison.csv', index=False)
        pd.DataFrame(performance_comparison).to_csv(f'{output_dir}/performance_comparison.csv', index=False)
        
        return config_comparison, performance_comparison
    
    def create_comparison_plots(self, app1_info, app2_info, app1_results, app2_results, output_dir):
        """Create comprehensive comparison visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Performance Comparison Bar Chart
        metrics = ['Test Accuracy', 'Test F1', 'Balanced Acc', 'Precision', 'Recall']
        app1_values = [
            app1_results['test_accuracy'], app1_results['test_f1_weighted'],
            app1_results['test_balanced_accuracy'], app1_results['test_precision'],
            app1_results['test_recall']
        ]
        app2_values = [
            app2_results['test_accuracy'], app2_results['test_f1_weighted'],
            app2_results['test_balanced_accuracy'], app2_results['test_precision'],
            app2_results['test_recall']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = axes[0,0].bar(x - width/2, app1_values, width, label='Approach 1', color='skyblue', alpha=0.8)
        bars2 = axes[0,0].bar(x + width/2, app2_values, width, label='Approach 2', color='lightcoral', alpha=0.8)
        
        axes[0,0].set_title('Performance Comparison', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Metrics')
        axes[0,0].set_ylabel('Score (%)')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(metrics, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0,0].annotate(f'{height:.1f}%',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=10)
        
        # 2. Generalization Analysis
        gap_metrics = ['Generalization Gap', 'Accuracy Gap', 'F1 Gap']
        app1_gaps = [app1_info['generalization_gap'], app1_info['accuracy_gap'], app1_info['f1_gap']]
        app2_gaps = [app2_info['generalization_gap'], app2_info['accuracy_gap'], app2_info['f1_gap']]
        
        x_gaps = np.arange(len(gap_metrics))
        bars1_gap = axes[0,1].bar(x_gaps - width/2, app1_gaps, width, label='Approach 1', color='orange', alpha=0.8)
        bars2_gap = axes[0,1].bar(x_gaps + width/2, app2_gaps, width, label='Approach 2', color='green', alpha=0.8)
        
        # Add target lines
        axes[0,1].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Gen Gap Limit (5%)')
        axes[0,1].axhline(y=2, color='blue', linestyle='--', alpha=0.7, label='Gap Target Min (2%)')
        
        axes[0,1].set_title('Generalization Analysis', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Gap Types')
        axes[0,1].set_ylabel('Gap (%)')
        axes[0,1].set_xticks(x_gaps)
        axes[0,1].set_xticklabels(gap_metrics)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1_gap, bars2_gap]:
            for bar in bars:
                height = bar.get_height()
                axes[0,1].annotate(f'{height:.1f}%',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=10)
        
        # 3. Model Efficiency Comparison
        efficiency_metrics = ['Model Size (MB)', 'Inference Time (ms)', 'Parameters (M)']
        app1_efficiency = [
            app1_info['model_size_mb'],
            app1_results['avg_inference_time'],
            app1_info['total_params'] / 1e6
        ]
        app2_efficiency = [
            app2_info['model_size_mb'],
            app2_results['avg_inference_time'],
            app2_info['total_params'] / 1e6
        ]
        
        x_eff = np.arange(len(efficiency_metrics))
        bars1_eff = axes[0,2].bar(x_eff - width/2, app1_efficiency, width, label='Approach 1', color='purple', alpha=0.8)
        bars2_eff = axes[0,2].bar(x_eff + width/2, app2_efficiency, width, label='Approach 2', color='teal', alpha=0.8)
        
        # Add target lines
        axes[0,2].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Mobile Size Limit (5MB)')
        axes[0,2].axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Target Inference (25ms)')
        
        axes[0,2].set_title('Model Efficiency', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Efficiency Metrics')
        axes[0,2].set_ylabel('Value')
        axes[0,2].set_xticks(x_eff)
        axes[0,2].set_xticklabels(efficiency_metrics)
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Training Progress Comparison
        if app1_info['history'] and app2_info['history']:
            # Plot training curves
            app1_epochs = app1_info['history'].get('epochs', [])
            app1_val_acc = app1_info['history'].get('val_acc', [])
            app2_epochs = app2_info['history'].get('epochs', [])
            app2_val_acc = app2_info['history'].get('val_acc', [])
            
            if app1_epochs and app2_epochs:
                axes[1,0].plot(app1_epochs, app1_val_acc, 'b-', label='Approach 1', linewidth=2)
                axes[1,0].plot(app2_epochs, app2_val_acc, 'r-', label='Approach 2', linewidth=2)
                axes[1,0].set_title('Validation Accuracy Progress', fontsize=14, fontweight='bold')
                axes[1,0].set_xlabel('Epoch')
                axes[1,0].set_ylabel('Validation Accuracy (%)')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
        
        # 5. Confusion Matrix Comparison
        if 'confusion_matrix' in app1_results and 'confusion_matrix' in app2_results:
            # Approach 1 confusion matrix
            im1 = axes[1,1].imshow(app1_results['confusion_matrix'], cmap='Blues', alpha=0.8)
            axes[1,1].set_title('Approach 1 - Confusion Matrix', fontsize=12, fontweight='bold')
            
            # Add text annotations
            cm1 = app1_results['confusion_matrix']
            for i in range(cm1.shape[0]):
                for j in range(cm1.shape[1]):
                    axes[1,1].text(j, i, str(cm1[i, j]), ha='center', va='center')
            
            # Approach 2 confusion matrix
            im2 = axes[1,2].imshow(app2_results['confusion_matrix'], cmap='Reds', alpha=0.8)
            axes[1,2].set_title('Approach 2 - Confusion Matrix', fontsize=12, fontweight='bold')
            
            # Add text annotations
            cm2 = app2_results['confusion_matrix']
            for i in range(cm2.shape[0]):
                for j in range(cm2.shape[1]):
                    axes[1,2].text(j, i, str(cm2[i, j]), ha='center', va='center')
        
        plt.suptitle('MobileNetV1 1D ECG Classification - Approach Comparison\n' +
                    f'Approach 1: {app1_results["test_accuracy"]:.2f}% Accuracy | ' +
                    f'Approach 2: {app2_results["test_accuracy"]:.2f}% Accuracy',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_deployment_comparison(self, app1_info, app2_info, app1_results, app2_results, output_dir):
        """Create deployment readiness comparison"""
        
        deployment_comparison = {
            'approach_1_performance_first': {
                'model_info': {
                    'name': 'Approach 1 (Performance-First)',
                    'epoch': int(app1_info['epoch']),
                    'model_size_mb': float(app1_info['model_size_mb']),
                    'total_parameters': int(app1_info['total_params']),
                    'width_multiplier': float(app1_info['config'].get('width_multiplier', 0))
                },
                'performance': {
                    'test_accuracy': float(app1_results['test_accuracy']),
                    'test_f1_weighted': float(app1_results['test_f1_weighted']),
                    'test_f1_macro': float(app1_results['test_f1_macro']),
                    'balanced_accuracy': float(app1_results['test_balanced_accuracy']),
                    'generalization_gap': float(app1_info['generalization_gap']),
                    'accuracy_gap': float(app1_info['accuracy_gap']),
                    'f1_gap': float(app1_info['f1_gap'])
                },
                'deployment': {
                    'avg_inference_ms': float(app1_results['avg_inference_time']),
                    'real_time_ready': bool(app1_results['avg_inference_time'] < 50),
                    'mobile_ready': bool(app1_info['model_size_mb'] < 5),
                    'clinical_grade': bool(app1_results['test_accuracy'] > 90),
                    'deployment_recommendation': self.get_deployment_recommendation(app1_info, app1_results)
                }
            },
            'approach_2_generalization_first': {
                'model_info': {
                    'name': 'Approach 2 (Generalization-First)',
                    'epoch': int(app2_info['epoch']),
                    'model_size_mb': float(app2_info['model_size_mb']),
                    'total_parameters': int(app2_info['total_params']),
                    'width_multiplier': float(app2_info['config'].get('width_multiplier', 0))
                },
                'performance': {
                    'test_accuracy': float(app2_results['test_accuracy']),
                    'test_f1_weighted': float(app2_results['test_f1_weighted']),
                    'test_f1_macro': float(app2_results['test_f1_macro']),
                    'balanced_accuracy': float(app2_results['test_balanced_accuracy']),
                    'generalization_gap': float(app2_info['generalization_gap']),
                    'accuracy_gap': float(app2_info['accuracy_gap']),
                    'f1_gap': float(app2_info['f1_gap'])
                },
                'deployment': {
                    'avg_inference_ms': float(app2_results['avg_inference_time']),
                    'real_time_ready': bool(app2_results['avg_inference_time'] < 50),
                    'mobile_ready': bool(app2_info['model_size_mb'] < 5),
                    'clinical_grade': bool(app2_results['test_accuracy'] > 90),
                    'deployment_recommendation': self.get_deployment_recommendation(app2_info, app2_results)
                }
            },
            'comparison_summary': {
                'winner_accuracy': 'Approach 1' if app1_results['test_accuracy'] > app2_results['test_accuracy'] else 'Approach 2',
                'winner_generalization': 'Approach 1' if app1_info['generalization_gap'] < app2_info['generalization_gap'] else 'Approach 2',
                'winner_speed': 'Approach 1' if app1_results['avg_inference_time'] < app2_results['avg_inference_time'] else 'Approach 2',
                'winner_size': 'Approach 1' if app1_info['model_size_mb'] < app2_info['model_size_mb'] else 'Approach 2',
                'overall_recommendation': self.get_overall_recommendation(app1_info, app2_info, app1_results, app2_results)
            },
            'generated_at': datetime.now().isoformat()
        }
        
        with open(f'{output_dir}/deployment_comparison.json', 'w') as f:
            json.dump(deployment_comparison, f, indent=2)
        
        return deployment_comparison
    
    def get_deployment_recommendation(self, model_info, results):
        """Get deployment recommendation for a single model"""
        score = 0
        
        # Performance scoring
        if results['test_accuracy'] > 95: score += 25
        elif results['test_accuracy'] > 90: score += 20
        elif results['test_accuracy'] > 85: score += 10
        
        # Generalization scoring
        if model_info['generalization_gap'] < 3: score += 25
        elif model_info['generalization_gap'] < 5: score += 20
        elif model_info['generalization_gap'] < 8: score += 10
        
        # Efficiency scoring
        if results['avg_inference_time'] < 25: score += 25
        elif results['avg_inference_time'] < 50: score += 15
        elif results['avg_inference_time'] < 100: score += 5
        
        # Size scoring
        if model_info['model_size_mb'] < 3: score += 25
        elif model_info['model_size_mb'] < 5: score += 20
        elif model_info['model_size_mb'] < 10: score += 10
        
        # Determine recommendation
        if score >= 85:
            return "EXCELLENT - Ready for production deployment"
        elif score >= 70:
            return "GOOD - Suitable for deployment with minor optimizations"
        elif score >= 50:
            return "ACCEPTABLE - Consider further optimization"
        else:
            return "NEEDS IMPROVEMENT - Significant optimization required"
    
    def get_overall_recommendation(self, app1_info, app2_info, app1_results, app2_results):
        """Get overall recommendation between approaches"""
        
        # Score both approaches
        app1_score = 0
        app2_score = 0
        
        # Accuracy comparison
        if app1_results['test_accuracy'] > app2_results['test_accuracy']:
            app1_score += 20
        else:
            app2_score += 20
        
        # Generalization comparison
        if app1_info['generalization_gap'] < app2_info['generalization_gap']:
            app1_score += 25
        else:
            app2_score += 25
        
        # Efficiency comparison
        if app1_results['avg_inference_time'] < app2_results['avg_inference_time']:
            app1_score += 15
        else:
            app2_score += 15
        
        # Size comparison
        if app1_info['model_size_mb'] < app2_info['model_size_mb']:
            app1_score += 15
        else:
            app2_score += 15
        
        # F1 comparison
        if app1_results['test_f1_weighted'] > app2_results['test_f1_weighted']:
            app1_score += 15
        else:
            app2_score += 15
        
        # Mobile readiness
        if app1_info['model_size_mb'] < 5 and app1_results['avg_inference_time'] < 50:
            app1_score += 10
        if app2_info['model_size_mb'] < 5 and app2_results['avg_inference_time'] < 50:
            app2_score += 10
        
        if app1_score > app2_score:
            return f"Approach 1 (Performance-First) - Score: {app1_score}/100"
        elif app2_score > app1_score:
            return f"Approach 2 (Generalization-First) - Score: {app2_score}/100"
        else:
            return f"TIE - Both approaches equally good - Scores: {app1_score}/100"
    
    def create_final_report(self, app1_info, app2_info, app1_results, app2_results, output_dir):
        """Create comprehensive markdown report"""
        
        report_content = f"""# MobileNetV1 1D ECG Classification - Model Comparison Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report compares two MobileNetV1 1D models trained for real-time ECG classification:

- **Approach 1**: Performance-First (Maximum accuracy focus)
- **Approach 2**: Generalization-First (Tight gap focus)

## Key Results Summary

| Metric | Approach 1 | Approach 2 | Winner |
|--------|------------|------------|--------|
| **Test Accuracy** | {app1_results['test_accuracy']:.2f}% | {app2_results['test_accuracy']:.2f}% | {"App 1" if app1_results['test_accuracy'] > app2_results['test_accuracy'] else "App 2"} |
| **Test F1 (Weighted)** | {app1_results['test_f1_weighted']:.2f}% | {app2_results['test_f1_weighted']:.2f}% | {"App 1" if app1_results['test_f1_weighted'] > app2_results['test_f1_weighted'] else "App 2"} |
| **Generalization Gap** | {app1_info['generalization_gap']:.2f}% | {app2_info['generalization_gap']:.2f}% | {"App 1" if app1_info['generalization_gap'] < app2_info['generalization_gap'] else "App 2"} |
| **Model Size** | {app1_info['model_size_mb']:.2f} MB | {app2_info['model_size_mb']:.2f} MB | {"App 1" if app1_info['model_size_mb'] < app2_info['model_size_mb'] else "App 2"} |
| **Inference Speed** | {app1_results['avg_inference_time']:.2f} ms | {app2_results['avg_inference_time']:.2f} ms | {"App 1" if app1_results['avg_inference_time'] < app2_results['avg_inference_time'] else "App 2"} |

## Detailed Analysis

### Approach 1: Performance-First
- **Strategy**: Maximize accuracy with acceptable generalization
- **Training Epoch**: {app1_info['epoch']}
- **Validation Accuracy**: {app1_info['val_acc']:.2f}%
- **Test Accuracy**: {app1_results['test_accuracy']:.2f}%
- **Generalization Gap**: {app1_info['generalization_gap']:.2f}%
- **Model Size**: {app1_info['model_size_mb']:.2f} MB
- **Parameters**: {app1_info['total_params']:,}
- **Inference Time**: {app1_results['avg_inference_time']:.2f} ms/sample

**Strengths**:
- {"High accuracy performance" if app1_results['test_accuracy'] > 93 else "Moderate accuracy performance"}
- {"Fast inference speed" if app1_results['avg_inference_time'] < 30 else "Acceptable inference speed"}
- {"Mobile-ready size" if app1_info['model_size_mb'] < 5 else "Large model size"}

**Weaknesses**:
- {"Acceptable generalization" if app1_info['generalization_gap'] < 8 else "Poor generalization"}
- {"Efficient training" if app1_info['epoch'] < 100 else "Long training time"}

### Approach 2: Generalization-First  
- **Strategy**: Tight generalization gaps with excellent accuracy
- **Training Epoch**: {app2_info['epoch']}
- **Validation Accuracy**: {app2_info['val_acc']:.2f}%
- **Test Accuracy**: {app2_results['test_accuracy']:.2f}%
- **Generalization Gap**: {app2_info['generalization_gap']:.2f}%
- **Model Size**: {app2_info['model_size_mb']:.2f} MB  
- **Parameters**: {app2_info['total_params']:,}
- **Inference Time**: {app2_results['avg_inference_time']:.2f} ms/sample

**Strengths**:
- {"Excellent generalization" if app2_info['generalization_gap'] < 5 else "Good generalization"}
- {"Tight gap control" if app2_info['accuracy_gap'] < 5 else "Loose gap control"}
- {"Mobile-ready size" if app2_info['model_size_mb'] < 5 else "Large model size"}
- {"Clinical-grade accuracy" if app2_results['test_accuracy'] > 90 else "Moderate accuracy"}

**Weaknesses**:
- {"Slightly slower inference" if app2_results['avg_inference_time'] > app1_results['avg_inference_time'] else "Similar inference speed"}

## Gap Analysis Detailed

### Generalization Gap Requirements: <5%
- **Approach 1**: {app1_info['generalization_gap']:.2f}% {"MEETS" if app1_info['generalization_gap'] < 5 else "EXCEEDS"}
- **Approach 2**: {app2_info['generalization_gap']:.2f}% {"MEETS" if app2_info['generalization_gap'] < 5 else "EXCEEDS"}

### Accuracy Gap Requirements: 2-5%  
- **Approach 1**: {app1_info['accuracy_gap']:.2f}% {"PERFECT" if 2 <= app1_info['accuracy_gap'] <= 5 else ("TOO TIGHT" if app1_info['accuracy_gap'] < 2 else "TOO WIDE")}
- **Approach 2**: {app2_info['accuracy_gap']:.2f}% {"PERFECT" if 2 <= app2_info['accuracy_gap'] <= 5 else ("TOO TIGHT" if app2_info['accuracy_gap'] < 2 else "TOO WIDE")}

### F1 Gap Requirements: 1-4%
- **Approach 1**: {app1_info['f1_gap']:.2f}% {"PERFECT" if 1 <= app1_info['f1_gap'] <= 4 else ("TOO TIGHT" if app1_info['f1_gap'] < 1 else "TOO WIDE")}
- **Approach 2**: {app2_info['f1_gap']:.2f}% {"PERFECT" if 1 <= app2_info['f1_gap'] <= 4 else ("TOO TIGHT" if app2_info['f1_gap'] < 1 else "TOO WIDE")}

## Mobile Deployment Analysis

### Size Requirements (<5MB for mobile apps):
- **Approach 1**: {app1_info['model_size_mb']:.2f} MB {"MOBILE READY" if app1_info['model_size_mb'] < 5 else "TOO LARGE"}
- **Approach 2**: {app2_info['model_size_mb']:.2f} MB {"MOBILE READY" if app2_info['model_size_mb'] < 5 else "TOO LARGE"}

### Speed Requirements (<50ms for real-time):
- **Approach 1**: {app1_results['avg_inference_time']:.2f} ms {"REAL-TIME" if app1_results['avg_inference_time'] < 50 else "TOO SLOW"}
- **Approach 2**: {app2_results['avg_inference_time']:.2f} ms {"REAL-TIME" if app2_results['avg_inference_time'] < 50 else "TOO SLOW"}

## Clinical Performance Analysis

### Per-Class Performance Comparison:

#### Approach 1 - Classification Breakdown:
```
Class Performance:
{self.format_class_performance(app1_results.get('per_class_metrics', {}))}
```

#### Approach 2 - Classification Breakdown:
```
Class Performance:
{self.format_class_performance(app2_results.get('per_class_metrics', {}))}
```

## Deployment Recommendations

### For Production Deployment:
**Recommended Model**: {self.get_overall_recommendation(app1_info, app2_info, app1_results, app2_results)}

### Use Case Specific Recommendations:

#### **Mobile/Smartphone Apps**:
- **Best Choice**: {"Approach 2" if app2_info['model_size_mb'] < app1_info['model_size_mb'] and app2_results['avg_inference_time'] < 50 else "Approach 1"}
- **Reason**: {"Smaller size and good performance" if app2_info['model_size_mb'] < 5 else "Better balance of size and performance"}

#### **Clinical Decision Support**:
- **Best Choice**: {"Approach 1" if app1_results['test_accuracy'] > app2_results['test_accuracy'] else "Approach 2"}
- **Reason**: {"Highest accuracy for critical decisions" if app1_results['test_accuracy'] > app2_results['test_accuracy'] else "Best generalization for diverse patients"}

#### **Wearable Devices**:
- **Best Choice**: {"Approach 2" if app2_info['model_size_mb'] < 3 and app2_results['avg_inference_time'] < 30 else "Approach 1"}
- **Reason**: {"Ultra-compact with good performance" if app2_info['model_size_mb'] < 3 else "Best available efficiency"}

#### **Research/Benchmarking**:
- **Best Choice**: {"Approach 1" if app1_results['test_f1_weighted'] > app2_results['test_f1_weighted'] else "Approach 2"}
- **Reason**: {"Highest F1 performance" if app1_results['test_f1_weighted'] > app2_results['test_f1_weighted'] else "Best generalization characteristics"}

## Training Efficiency Analysis

| Aspect | Approach 1 | Approach 2 |
|--------|------------|------------|
| **Training Time** | ~{int(app1_info['epoch']) * 10} minutes | ~{int(app2_info['epoch']) * 10} minutes |
| **Epochs to Best** | {app1_info['epoch']} epochs | {app2_info['epoch']} epochs |
| **Training Efficiency** | {app1_results['test_accuracy'] / app1_info['epoch']:.3f}% per epoch | {app2_results['test_accuracy'] / app2_info['epoch']:.3f}% per epoch |
| **Convergence Speed** | {"Fast" if app1_info['epoch'] < 50 else "Moderate" if app1_info['epoch'] < 100 else "Slow"} | {"Fast" if app2_info['epoch'] < 50 else "Moderate" if app2_info['epoch'] < 100 else "Slow"} |

## Technical Deep Dive

### Configuration Differences:

| Parameter | Approach 1 | Approach 2 | Impact |
|-----------|------------|------------|--------|
| **Batch Size** | {app1_info['config'].get('batch_size', 'N/A')} | {app2_info['config'].get('batch_size', 'N/A')} | {"Larger batches favor speed" if app1_info['config'].get('batch_size', 0) > app2_info['config'].get('batch_size', 0) else "Smaller batches favor generalization"} |
| **Learning Rate** | {app1_info['config'].get('learning_rate', 'N/A')} | {app2_info['config'].get('learning_rate', 'N/A')} | {"Higher LR for faster learning" if app1_info['config'].get('learning_rate', 0) > app2_info['config'].get('learning_rate', 0) else "Lower LR for stability"} |
| **Dropout** | {app1_info['config'].get('dropout_rate', 'N/A')} | {app2_info['config'].get('dropout_rate', 'N/A')} | {"Higher dropout for regularization" if app2_info['config'].get('dropout_rate', 0) > app1_info['config'].get('dropout_rate', 0) else "Lower dropout for performance"} |
| **Width Multiplier** | {app1_info['config'].get('width_multiplier', 'N/A')} | {app2_info['config'].get('width_multiplier', 'N/A')} | {"Larger model for accuracy" if app1_info['config'].get('width_multiplier', 0) > app2_info['config'].get('width_multiplier', 0) else "Smaller model for efficiency"} |

## Final Verdict

### **WINNER ANALYSIS**:

**Accuracy Winner**: {"Approach 1" if app1_results['test_accuracy'] > app2_results['test_accuracy'] else "Approach 2"} ({max(app1_results['test_accuracy'], app2_results['test_accuracy']):.2f}%)

**Generalization Winner**: {"Approach 1" if app1_info['generalization_gap'] < app2_info['generalization_gap'] else "Approach 2"} ({min(app1_info['generalization_gap'], app2_info['generalization_gap']):.2f}% gap)

**Efficiency Winner**: {"Approach 1" if app1_info['model_size_mb'] < app2_info['model_size_mb'] else "Approach 2"} ({min(app1_info['model_size_mb'], app2_info['model_size_mb']):.2f} MB)

**Speed Winner**: {"Approach 1" if app1_results['avg_inference_time'] < app2_results['avg_inference_time'] else "Approach 2"} ({min(app1_results['avg_inference_time'], app2_results['avg_inference_time']):.2f} ms)

### **DEPLOYMENT RECOMMENDATION**:

{self.get_overall_recommendation(app1_info, app2_info, app1_results, app2_results)}

### **Mobile Deployment Status**:

| Approach | Mobile Ready | Real-time Ready | Clinical Grade | Overall Status |
|----------|--------------|-----------------|----------------|----------------|
| **Approach 1** | {"YES" if app1_info['model_size_mb'] < 5 else "NO"} | {"YES" if app1_results['avg_inference_time'] < 50 else "NO"} | {"YES" if app1_results['test_accuracy'] > 90 else "NO"} | {self.get_deployment_recommendation(app1_info, app1_results)} |
| **Approach 2** | {"YES" if app2_info['model_size_mb'] < 5 else "NO"} | {"YES" if app2_results['avg_inference_time'] < 50 else "NO"} | {"YES" if app2_results['test_accuracy'] > 90 else "NO"} | {self.get_deployment_recommendation(app2_info, app2_results)} |

## Gap Requirements Analysis

Your requirements: **Gen<5%, Acc=2-5%, F1=1-4%**

### Approach 1 Gap Compliance:
- **Generalization**: {app1_info['generalization_gap']:.2f}% {"PASS" if app1_info['generalization_gap'] < 5 else "FAIL"} (Target: <5%)
- **Accuracy**: {app1_info['accuracy_gap']:.2f}% {"PASS" if 2 <= app1_info['accuracy_gap'] <= 5 else "FAIL"} (Target: 2-5%)
- **F1**: {app1_info['f1_gap']:.2f}% {"PASS" if 1 <= app1_info['f1_gap'] <= 4 else "FAIL"} (Target: 1-4%)

### Approach 2 Gap Compliance:
- **Generalization**: {app2_info['generalization_gap']:.2f}% {"PASS" if app2_info['generalization_gap'] < 5 else "FAIL"} (Target: <5%)
- **Accuracy**: {app2_info['accuracy_gap']:.2f}% {"PASS" if 2 <= app2_info['accuracy_gap'] <= 5 else "FAIL"} (Target: 2-5%)
- **F1**: {app2_info['f1_gap']:.2f}% {"PASS" if 1 <= app2_info['f1_gap'] <= 4 else "FAIL"} (Target: 1-4%)

## Key Insights

1. **Performance vs Generalization Trade-off**: {"Approach 1 achieved higher accuracy but with wider gaps" if app1_results['test_accuracy'] > app2_results['test_accuracy'] and app1_info['generalization_gap'] > app2_info['generalization_gap'] else "Both approaches achieved similar performance"}

2. **Training Efficiency**: {"Approach 1 required more epochs but achieved higher accuracy" if app1_info['epoch'] > app2_info['epoch'] and app1_results['test_accuracy'] > app2_results['test_accuracy'] else "Approach 2 achieved good results with fewer epochs"}

3. **Mobile Readiness**: {"Both models are mobile-ready" if app1_info['model_size_mb'] < 5 and app2_info['model_size_mb'] < 5 else ("Only Approach 2 is mobile-ready" if app2_info['model_size_mb'] < 5 else "Neither model meets mobile size requirements")}

## Final Recommendation

Based on the comprehensive analysis:

**For Production Deployment**: Use **{self.get_overall_recommendation(app1_info, app2_info, app1_results, app2_results).split(' - ')[0]}**

**Reasoning**:
- Performance requirements are {"met by both" if min(app1_results['test_accuracy'], app2_results['test_accuracy']) > 90 else "challenging"}
- Generalization requirements are {"excellently met" if max(app1_info['generalization_gap'], app2_info['generalization_gap']) < 5 else "partially met"}
- Mobile requirements are {"satisfied" if min(app1_info['model_size_mb'], app2_info['model_size_mb']) < 5 else "need optimization"}

---

**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Analysis Tool**: MobileNetV1 1D Model Comparison Suite
**Data**: ECG Classification (5 classes: F, N, Q, S, V)
"""

        # Save report
        with open(f'{output_dir}/comparison_report.md', 'w') as f:
            f.write(report_content)
        
        return report_content
    
    def format_class_performance(self, per_class_metrics):
        """Format per-class performance metrics"""
        if not per_class_metrics:
            return "Per-class metrics not available"
        
        formatted = ""
        for class_id, metrics in per_class_metrics.items():
            if class_id in ['0', '1', '2', '3', '4']:  # Only class indices
                class_names = {'0': 'F', '1': 'N', '2': 'Q', '3': 'S', '4': 'V'}
                class_name = class_names.get(class_id, class_id)
                
                precision = metrics.get('precision', 0) * 100
                recall = metrics.get('recall', 0) * 100
                f1 = metrics.get('f1-score', 0) * 100
                support = metrics.get('support', 0)
                
                formatted += f"   {class_name}: Precision={precision:.1f}%, Recall={recall:.1f}%, F1={f1:.1f}%, Support={support}\n"
        
        return formatted.strip()

    def generate_three_way_comparison(self, models_info, output_dir='outputs/comparison'):
        """Generate comprehensive 3-way comparison report"""
        print(f"\n三-WAY MODEL COMPARISON REPORT")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate all models on test set
        results = {}
        for name, info in models_info.items():
            if info:
                results[name] = self.evaluate_model_on_test_set(info)
            else:
                print(f"Skipping {name} - model not loaded properly")
        
        if len(results) < 2:
            print("Need at least 2 models for comparison")
            return
        
        # Generate 3-way comparison
        self.create_three_way_tables(models_info, results, output_dir)
        self.create_three_way_plots(models_info, results, output_dir)
        self.create_three_way_report(models_info, results, output_dir)
        
        print(f"\n3-WAY COMPARISON REPORT GENERATED!")
        print(f"Location: {output_dir}/")
    
    def create_three_way_tables(self, models_info, results, output_dir):
        """Create 3-way comparison tables"""
        
        # Extract model names
        model_names = list(models_info.keys())
        
        # Performance comparison table
        metrics = [
            'Test Accuracy (%)', 'Test F1 Weighted (%)', 'Test F1 Macro (%)',
            'Balanced Accuracy (%)', 'Generalization Gap (%)', 'Accuracy Gap (%)',
            'F1 Gap (%)', 'Model Size (MB)', 'Inference Time (ms)', 
            'Total Parameters', 'Training Epoch'
        ]
        
        comparison_data = {'Metric': metrics}
        
        for name in model_names:
            if name in results and name in models_info:
                info = models_info[name]
                result = results[name]
                
                comparison_data[name] = [
                    f"{result['test_accuracy']:.2f}",
                    f"{result['test_f1_weighted']:.2f}",
                    f"{result['test_f1_macro']:.2f}",
                    f"{result['test_balanced_accuracy']:.2f}",
                    f"{info['generalization_gap']:.2f}",
                    f"{info['accuracy_gap']:.2f}",
                    f"{info['f1_gap']:.2f}",
                    f"{info['model_size_mb']:.2f}",
                    f"{result['avg_inference_time']:.2f}",
                    f"{info['total_params']:,}",
                    f"{info['epoch']}"
                ]
        
        # Save comparison table
        df = pd.DataFrame(comparison_data)
        df.to_csv(f'{output_dir}/three_way_comparison.csv', index=False)
        
        return comparison_data
    
    def create_three_way_plots(self, models_info, results, output_dir):
        """Create 3-way comparison visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        model_names = list(models_info.keys())
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'mediumpurple'][:len(model_names)]
        
        # 1. Performance Comparison
        performance_metrics = ['Test Accuracy', 'Test F1', 'Balanced Acc']
        performance_data = []
        
        for name in model_names:
            if name in results:
                performance_data.append([
                    results[name]['test_accuracy'],
                    results[name]['test_f1_weighted'],
                    results[name]['test_balanced_accuracy']
                ])
        
        x = np.arange(len(performance_metrics))
        width = 0.8 / len(model_names)  # Dynamic width based on number of models
        
        for i, (name, data) in enumerate(zip(model_names, performance_data)):
            offset = (i - len(model_names)//2) * width
            bars = axes[0,0].bar(x + offset, data, width, label=name, color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[0,0].annotate(f'{height:.1f}%',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha='center', va='bottom', fontsize=9)
        
        axes[0,0].set_title('Performance Comparison', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Metrics')
        axes[0,0].set_ylabel('Score (%)')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(performance_metrics)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Gap Analysis
        gap_metrics = ['Gen Gap', 'Acc Gap', 'F1 Gap']
        gap_data = []
        
        for name in model_names:
            if name in models_info:
                gap_data.append([
                    models_info[name]['generalization_gap'],
                    models_info[name]['accuracy_gap'],
                    models_info[name]['f1_gap']
                ])
        
        x_gap = np.arange(len(gap_metrics))
        for i, (name, data) in enumerate(zip(model_names, gap_data)):
            offset = (i - len(model_names)//2) * width
            bars = axes[0,1].bar(x_gap + offset, data, width, label=name, color=colors[i], alpha=0.8)
        
        # Add target lines
        axes[0,1].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Gen Gap Limit (5%)')
        axes[0,1].axhline(y=2, color='blue', linestyle='--', alpha=0.7, label='Min Gap Target (2%)')
        
        axes[0,1].set_title('Gap Analysis', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Gap Types')
        axes[0,1].set_ylabel('Gap (%)')
        axes[0,1].set_xticks(x_gap)
        axes[0,1].set_xticklabels(gap_metrics)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Efficiency Comparison
        efficiency_metrics = ['Size (MB)', 'Speed (ms)', 'Params (M)']
        efficiency_data = []
        
        for name in model_names:
            if name in models_info and name in results:
                efficiency_data.append([
                    models_info[name]['model_size_mb'],
                    results[name]['avg_inference_time'],
                    models_info[name]['total_params'] / 1e6
                ])
        
        x_eff = np.arange(len(efficiency_metrics))
        for i, (name, data) in enumerate(zip(model_names, efficiency_data)):
            offset = (i - len(model_names)//2) * width
            bars = axes[0,2].bar(x_eff + offset, data, width, label=name, color=colors[i], alpha=0.8)
        
        axes[0,2].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Mobile Limit (5MB)')
        axes[0,2].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Real-time Limit (50ms)')
        
        axes[0,2].set_title('Efficiency Comparison', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Efficiency Metrics')
        axes[0,2].set_ylabel('Value')
        axes[0,2].set_xticks(x_eff)
        axes[0,2].set_xticklabels(efficiency_metrics)
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Training Progress (if available)
        for i, name in enumerate(model_names):
            if name in models_info and 'history' in models_info[name]:
                history = models_info[name]['history']
                if 'epochs' in history and 'val_acc' in history:
                    epochs = history['epochs']
                    val_acc = history['val_acc']
                    axes[1,0].plot(epochs, val_acc, color=colors[i], label=name, linewidth=2)
        
        axes[1,0].set_title('Training Progress Comparison', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Validation Accuracy (%)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Gap Requirements Compliance
        gap_requirements = ['Gen<5%', 'Acc 2-5%', 'F1 1-4%']
        compliance_data = []
        
        for name in model_names:
            if name in models_info:
                info = models_info[name]
                compliance = [
                    1 if info['generalization_gap'] < 5 else 0,
                    1 if 2 <= info['accuracy_gap'] <= 5 else 0,
                    1 if 1 <= info['f1_gap'] <= 4 else 0
                ]
                compliance_data.append(compliance)
        
        x_req = np.arange(len(gap_requirements))
        for i, (name, data) in enumerate(zip(model_names, compliance_data)):
            offset = (i - len(model_names)//2) * width
            bars = axes[1,1].bar(x_req + offset, data, width, label=name, color=colors[i], alpha=0.8)
            
            # Add checkmarks for compliance
            for j, bar in enumerate(bars):
                height = bar.get_height()
                symbol = "PASS" if height == 1 else "FAIL"
                axes[1,1].annotate(symbol,
                                 xy=(bar.get_x() + bar.get_width() / 2, height/2),
                                 ha='center', va='center', fontsize=10)
        
        axes[1,1].set_title('Gap Requirements Compliance', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Requirements')
        axes[1,1].set_ylabel('Compliance (1=Yes, 0=No)')
        axes[1,1].set_xticks(x_req)
        axes[1,1].set_xticklabels(gap_requirements)
        axes[1,1].set_ylim(0, 1.2)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Overall Score Comparison
        score_categories = ['Accuracy', 'Generalization', 'Speed', 'Size', 'Clinical']
        
        for i, name in enumerate(model_names):
            if name in models_info and name in results:
                info = models_info[name]
                result = results[name]
                
                # Calculate scores (0-100)
                accuracy_score = min(100, result['test_accuracy'])
                gen_score = max(0, 100 - info['generalization_gap'] * 10)
                speed_score = max(0, 100 - result['avg_inference_time'] * 2)
                size_score = max(0, 100 - info['model_size_mb'] * 15)
                clinical_score = min(100, result['test_f1_weighted'])
                
                scores = [accuracy_score, gen_score, speed_score, size_score, clinical_score]
                
                offset = (i - len(model_names)//2) * width
                bars = axes[1,2].bar(np.arange(len(score_categories)) + offset, scores, 
                                   width, label=name, color=colors[i], alpha=0.8)
        
        axes[1,2].set_title('Overall Score Comparison', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Score Categories')
        axes[1,2].set_ylabel('Score (0-100)')
        axes[1,2].set_xticks(np.arange(len(score_categories)))
        axes[1,2].set_xticklabels(score_categories)
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.suptitle('MobileNetV1 1D ECG Classification - Model Comparison\n' +
                    f'Models: {", ".join(model_names)}',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/three_way_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_three_way_report(self, models_info, results, output_dir):
        """Create comprehensive 3-way markdown report"""
        
        model_names = list(models_info.keys())
        
        report_content = f"""# MobileNetV1 1D ECG Classification - Multi-Model Comparison Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report compares **{len(model_names)} MobileNetV1 1D models** trained for real-time ECG classification:

"""
        
        for i, name in enumerate(model_names, 1):
            if name in models_info:
                info = models_info[name]
                result = results.get(name, {})
                report_content += f"- **Model {i}**: {name} (Epoch {info['epoch']}, {result.get('test_accuracy', 0):.1f}% accuracy)\n"
        
        report_content += f"""

## Complete Performance Comparison

| Metric | {' | '.join(model_names)} |
|--------|{'|'.join(['--------'] * len(model_names))}|
"""
        
        # Add performance rows
        metrics_data = [
            ('Test Accuracy (%)', lambda r: f"{r.get('test_accuracy', 0):.2f}"),
            ('Test F1 Weighted (%)', lambda r: f"{r.get('test_f1_weighted', 0):.2f}"),
            ('Test F1 Macro (%)', lambda r: f"{r.get('test_f1_macro', 0):.2f}"),
            ('Balanced Accuracy (%)', lambda r: f"{r.get('test_balanced_accuracy', 0):.2f}"),
            ('Precision (%)', lambda r: f"{r.get('test_precision', 0):.2f}"),
            ('Recall (%)', lambda r: f"{r.get('test_recall', 0):.2f}"),
        ]
        
        for metric_name, formatter in metrics_data:
            row_data = [formatter(results.get(name, {})) for name in model_names]
            report_content += f"| **{metric_name}** | {' | '.join(row_data)} |\n"
        
        report_content += f"""

## Gap Analysis (Requirements: Gen<5%, Acc=2-5%, F1=1-4%)

| Gap Type | {' | '.join(model_names)} |
|----------|{'|'.join(['--------'] * len(model_names))}|
"""
        
        gap_metrics_data = [
            ('Generalization Gap (%)', 'generalization_gap', 5, '<'),
            ('Accuracy Gap (%)', 'accuracy_gap', (2, 5), 'range'),
            ('F1 Gap (%)', 'f1_gap', (1, 4), 'range')
        ]
        
        for metric_name, key, target, check_type in gap_metrics_data:
            row_data = []
            for name in model_names:
                if name in models_info:
                    value = models_info[name].get(key, 0)
                    if check_type == '<':
                        status = "PASS" if value < target else "FAIL"
                    else:  # range check
                        status = "PASS" if target[0] <= value <= target[1] else "FAIL"
                    row_data.append(f"{value:.2f}% {status}")
                else:
                    row_data.append("N/A")
            
            report_content += f"| **{metric_name}** | {' | '.join(row_data)} |\n"
        
        report_content += f"""

## Mobile Deployment Analysis

| Deployment Metric | {' | '.join(model_names)} |
|-------------------|{'|'.join(['--------'] * len(model_names))}|
"""
        
        deployment_metrics = [
            ('Model Size (MB)', lambda info, res: f"{info.get('model_size_mb', 0):.2f} {'PASS' if info.get('model_size_mb', 0) < 5 else 'FAIL'}"),
            ('Inference Time (ms)', lambda info, res: f"{res.get('avg_inference_time', 0):.2f} {'PASS' if res.get('avg_inference_time', 0) < 50 else 'FAIL'}"),
            ('Parameters', lambda info, res: f"{info.get('total_params', 0):,}"),
            ('Mobile Ready', lambda info, res: "YES" if info.get('model_size_mb', 0) < 5 and res.get('avg_inference_time', 0) < 50 else "NO"),
            ('Real-time Ready', lambda info, res: "YES" if res.get('avg_inference_time', 0) < 50 else "NO"),
        ]
        
        for metric_name, formatter in deployment_metrics:
            row_data = []
            for name in model_names:
                if name in models_info and name in results:
                    row_data.append(formatter(models_info[name], results[name]))
                else:
                    row_data.append("N/A")
            report_content += f"| **{metric_name}** | {' | '.join(row_data)} |\n"
        
        report_content += f"""

## Clinical Performance Breakdown

### Per-Class Performance Summary:

"""
        
        for name in model_names:
            if name in results and 'per_class_metrics' in results[name]:
                report_content += f"#### {name}:\n"
                metrics = results[name]['per_class_metrics']
                
                class_names = {'0': 'F (Fusion)', '1': 'N (Normal)', '2': 'Q (Unknown)', '3': 'S (Supraventricular)', '4': 'V (Ventricular)'}
                
                for class_id in ['0', '1', '2', '3', '4']:
                    if class_id in metrics:
                        class_name = class_names.get(class_id, f"Class {class_id}")
                        precision = metrics[class_id].get('precision', 0) * 100
                        recall = metrics[class_id].get('recall', 0) * 100
                        f1 = metrics[class_id].get('f1-score', 0) * 100
                        support = metrics[class_id].get('support', 0)
                        
                        report_content += f"- **{class_name}**: Precision={precision:.1f}%, Recall={recall:.1f}%, F1={f1:.1f}%, Support={support}\n"
                
                report_content += "\n"
        
        report_content += f"""

## Deployment Recommendations

### Overall Winner Analysis:

"""
        
        # Determine winners in each category
        winners = {}
        
        # Accuracy winner
        best_acc = max(results[name]['test_accuracy'] for name in model_names if name in results)
        acc_winner = [name for name in model_names if name in results and results[name]['test_accuracy'] == best_acc][0]
        winners['accuracy'] = (acc_winner, best_acc)
        
        # Generalization winner (lowest gap)
        best_gen = min(models_info[name]['generalization_gap'] for name in model_names if name in models_info)
        gen_winner = [name for name in model_names if name in models_info and models_info[name]['generalization_gap'] == best_gen][0]
        winners['generalization'] = (gen_winner, best_gen)
        
        # Speed winner
        best_speed = min(results[name]['avg_inference_time'] for name in model_names if name in results)
        speed_winner = [name for name in model_names if name in results and results[name]['avg_inference_time'] == best_speed][0]
        winners['speed'] = (speed_winner, best_speed)
        
        # Size winner
        best_size = min(models_info[name]['model_size_mb'] for name in model_names if name in models_info)
        size_winner = [name for name in model_names if name in models_info and models_info[name]['model_size_mb'] == best_size][0]
        winners['size'] = (size_winner, best_size)
        
        report_content += f"""
**Accuracy Champion**: {winners['accuracy'][0]} ({winners['accuracy'][1]:.2f}%)
**Generalization Champion**: {winners['generalization'][0]} ({winners['generalization'][1]:.2f}% gap)
**Speed Champion**: {winners['speed'][0]} ({winners['speed'][1]:.2f}ms)
**Size Champion**: {winners['size'][0]} ({winners['size'][1]:.2f}MB)

### Mobile App Deployment:
"""
        
        # Find best mobile model
        mobile_candidates = []
        for name in model_names:
            if name in models_info and name in results:
                info = models_info[name]
                result = results[name]
                if info['model_size_mb'] < 5 and result['avg_inference_time'] < 50:
                    mobile_score = result['test_accuracy'] + (5 - info['generalization_gap']) * 2
                    mobile_candidates.append((name, mobile_score, info, result))
        
        if mobile_candidates:
            best_mobile = max(mobile_candidates, key=lambda x: x[1])
            report_content += f"""
**Recommended**: {best_mobile[0]}
- Size: {best_mobile[2]['model_size_mb']:.2f}MB (mobile ready)
- Speed: {best_mobile[3]['avg_inference_time']:.2f}ms (real-time)
- Accuracy: {best_mobile[3]['test_accuracy']:.2f}% (clinical grade)
- Gap: {best_mobile[2]['generalization_gap']:.2f}% (excellent generalization)
"""
        else:
            report_content += """
**Status**: No models meet full mobile requirements (<5MB + <50ms)
**Recommendation**: Consider further optimization or quantization
"""
        
        report_content += f"""

### Clinical Deployment:
"""
        
        # Find best clinical model
        clinical_candidates = []
        for name in model_names:
            if name in models_info and name in results:
                result = results[name]
                if result['test_accuracy'] > 90:
                    clinical_score = result['test_accuracy'] * 1.5 + result['test_f1_weighted'] * 0.5
                    clinical_candidates.append((name, clinical_score, models_info[name], result))
        
        if clinical_candidates:
            best_clinical = max(clinical_candidates, key=lambda x: x[1])
            report_content += f"""
**Recommended**: {best_clinical[0]}
- Accuracy: {best_clinical[3]['test_accuracy']:.2f}% (clinical grade)
- F1 Score: {best_clinical[3]['test_f1_weighted']:.2f}% (balanced performance)
- Balanced Accuracy: {best_clinical[3]['test_balanced_accuracy']:.2f}% (handles class imbalance)
- Generalization: {best_clinical[2]['generalization_gap']:.2f}% gap (reliability)
"""
        
        report_content += f"""

## Training Efficiency Analysis

| Model | Training Epoch | Time Estimate | Accuracy/Epoch | Efficiency Rating |
|-------|----------------|---------------|----------------|-------------------|
"""
        
        for name in model_names:
            if name in models_info and name in results:
                info = models_info[name]
                result = results[name]
                
                epoch = info['epoch']
                time_est = f"~{epoch * 10} min"
                acc_per_epoch = result['test_accuracy'] / epoch
                
                if acc_per_epoch > 2:
                    efficiency = "Excellent"
                elif acc_per_epoch > 1.5:
                    efficiency = "Good"
                elif acc_per_epoch > 1:
                    efficiency = "Moderate"
                else:
                    efficiency = "Poor"
                
                report_content += f"| {name} | {epoch} | {time_est} | {acc_per_epoch:.3f}% | {efficiency} |\n"
        
        report_content += f"""

## Gap Requirements Detailed Analysis

Your specific requirements: **Gen<5%, Acc=2-5%, F1=1-4%**

"""
        
        for name in model_names:
            if name in models_info:
                info = models_info[name]
                
                gen_gap = info['generalization_gap']
                acc_gap = info['accuracy_gap']
                f1_gap = info['f1_gap']
                
                # Check compliance
                gen_ok = "PASS" if gen_gap < 5 else "FAIL"
                acc_ok = "PASS" if 2 <= acc_gap <= 5 else ("Too tight" if acc_gap < 2 else "Too wide")
                f1_ok = "PASS" if 1 <= f1_gap <= 4 else ("Too tight" if f1_gap < 1 else "Too wide")
                
                overall_compliance = "FULLY COMPLIANT" if all([
                    gen_gap < 5, 2 <= acc_gap <= 5, 1 <= f1_gap <= 4
                ]) else "PARTIAL COMPLIANCE"
                
                report_content += f"""
### {name}:
- **Generalization Gap**: {gen_gap:.2f}% {gen_ok} (Target: <5%)
- **Accuracy Gap**: {acc_gap:.2f}% {acc_ok} (Target: 2-5%)
- **F1 Gap**: {f1_gap:.2f}% {f1_ok} (Target: 1-4%)
- **Overall**: {overall_compliance}

"""
        
        report_content += f"""

## Model Economics Analysis

| Model | Training Cost | Model Size | Deployment Cost | Overall TCO |
|-------|---------------|------------|-----------------|-------------|
"""
        
        for name in model_names:
            if name in models_info and name in results:
                info = models_info[name]
                result = results[name]
                
                # Estimate costs (relative)
                train_cost = "$" * min(5, max(1, info['epoch'] // 30))
                
                if info['model_size_mb'] < 3:
                    deploy_cost = "$"
                elif info['model_size_mb'] < 5:
                    deploy_cost = "$"
                else:
                    deploy_cost = "$$"
                
                size_rating = f"{info['model_size_mb']:.1f}MB"
                
                # Overall TCO (Total Cost of Ownership)
                total_epochs = info['epoch']
                if total_epochs < 50 and info['model_size_mb'] < 5:
                    tco = "LOW"
                elif total_epochs < 100 and info['model_size_mb'] < 8:
                    tco = "MODERATE"
                else:
                    tco = "HIGH"
                
                report_content += f"| {name} | {train_cost} | {size_rating} | {deploy_cost} | {tco} |\n"
        
        report_content += f"""

## Final Ranking & Recommendations

### Overall Model Ranking:

"""
        
        # Calculate comprehensive scores for ranking
        model_scores = []
        
        for name in model_names:
            if name in models_info and name in results:
                info = models_info[name]
                result = results[name]
                
                # Calculate comprehensive score (0-100)
                accuracy_score = min(100, result['test_accuracy'])
                gap_score = max(0, 100 - info['generalization_gap'] * 15)
                speed_score = max(0, 100 - result['avg_inference_time'] * 1.5)
                size_score = max(0, 100 - info['model_size_mb'] * 12)
                f1_score = min(100, result['test_f1_weighted'])
                
                # Gap compliance bonus
                gap_bonus = 0
                if info['generalization_gap'] < 5:
                    gap_bonus += 10
                if 2 <= info['accuracy_gap'] <= 5:
                    gap_bonus += 5
                if 1 <= info['f1_gap'] <= 4:
                    gap_bonus += 5
                
                total_score = (accuracy_score * 0.3 + gap_score * 0.25 + 
                             speed_score * 0.2 + size_score * 0.15 + 
                             f1_score * 0.1 + gap_bonus)
                
                model_scores.append((name, total_score, info, result))
        
        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, score, info, result) in enumerate(model_scores, 1):
            medal = "#1" if i == 1 else "#2" if i == 2 else f"#{i}"
            report_content += f"""
{medal}: **{name}** (Score: {score:.1f}/100)
- Test Accuracy: {result['test_accuracy']:.2f}%
- Generalization Gap: {info['generalization_gap']:.2f}%
- Model Size: {info['model_size_mb']:.2f}MB
- Inference: {result['avg_inference_time']:.2f}ms
- Gap Compliance: {"Full" if all([info['generalization_gap'] < 5, 2 <= info['accuracy_gap'] <= 5, 1 <= info['f1_gap'] <= 4]) else "Partial"}

"""
        
        report_content += f"""

### Use Case Specific Recommendations:

#### **For Mobile/Smartphone Apps**:
"""
        
        mobile_best = None
        for name, score, info, result in model_scores:
            if info['model_size_mb'] < 5 and result['avg_inference_time'] < 50:
                mobile_best = (name, info, result)
                break
        
        if mobile_best:
            report_content += f"""
**Recommended**: {mobile_best[0]}
- Mobile-ready size: {mobile_best[1]['model_size_mb']:.2f}MB
- Real-time speed: {mobile_best[2]['avg_inference_time']:.2f}ms
- Good accuracy: {mobile_best[2]['test_accuracy']:.2f}%
"""
        else:
            report_content += "No models meet mobile requirements. Consider quantization.\n"
        
        report_content += f"""

#### **For Clinical Decision Support**:
**Recommended**: {model_scores[0][0]}
- Highest overall score with best balance of accuracy and reliability

#### **For Wearable Devices**:
"""
        
        wearable_best = None
        for name, score, info, result in model_scores:
            if info['model_size_mb'] < 3 and result['avg_inference_time'] < 30:
                wearable_best = (name, info, result)
                break
        
        if wearable_best:
            report_content += f"**Recommended**: {wearable_best[0]} (Ultra-compact and fast)\n"
        else:
            # Find closest
            smallest_model = min(model_scores, key=lambda x: x[2]['model_size_mb'])
            report_content += f"**Recommended**: {smallest_model[0]} (Smallest available: {smallest_model[2]['model_size_mb']:.2f}MB)\n"
        
        report_content += f"""

## Key Insights & Lessons Learned

1. **Performance vs Generalization Trade-off**: 
   - {"Higher capacity models achieved better accuracy but potentially wider gaps" if len([m for m in model_scores if m[2]['model_size_mb'] > 4]) > 0 else "All models achieved good balance"}

2. **Training Efficiency**: 
   - Average epochs to convergence: {np.mean([info['epoch'] for info in models_info.values()]):.1f}
   - Most efficient: {min(model_scores, key=lambda x: x[2]['epoch'])[0]} ({min(info['epoch'] for info in models_info.values())} epochs)

3. **Gap Control Success**: 
   - Models meeting all gap requirements: {len([name for name in model_names if name in models_info and models_info[name]['generalization_gap'] < 5 and 2 <= models_info[name]['accuracy_gap'] <= 5 and 1 <= models_info[name]['f1_gap'] <= 4])}/{len(model_names)}

4. **Mobile Readiness**: 
   - Models ready for mobile deployment: {len([name for name in model_names if name in models_info and name in results and models_info[name]['model_size_mb'] < 5 and results[name]['avg_inference_time'] < 50])}/{len(model_names)}

## Final Deployment Decision

### **WINNER**: {model_scores[0][0]}

**Justification**:
- Highest comprehensive score: {model_scores[0][1]:.1f}/100
- {"Excellent accuracy: " + str(round(model_scores[0][3]['test_accuracy'], 1)) + "%" if model_scores[0][3]['test_accuracy'] > 93 else "Good accuracy: " + str(round(model_scores[0][3]['test_accuracy'], 1)) + "%"}
- {"Excellent generalization: " + str(round(model_scores[0][2]['generalization_gap'], 1)) + "% gap" if model_scores[0][2]['generalization_gap'] < 5 else "Acceptable generalization: " + str(round(model_scores[0][2]['generalization_gap'], 1)) + "% gap"}
- {"Mobile-ready" if model_scores[0][2]['model_size_mb'] < 5 else "Needs compression for mobile"}
- {"Real-time capable" if model_scores[0][3]['avg_inference_time'] < 50 else "May need optimization for real-time"}

### **Deployment Steps**:
1. Use model: `{model_scores[0][0].replace(' ', '_').lower()}/mobilenet_best.pth`
2. {"Apply quantization for mobile deployment" if model_scores[0][2]['model_size_mb'] > 3 else "Direct deployment ready"}
3. {"Implement real-time optimizations" if model_scores[0][3]['avg_inference_time'] > 25 else "Real-time performance ready"}
4. {"Monitor generalization in production" if model_scores[0][2]['generalization_gap'] > 3 else "Excellent generalization confidence"}

---

**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Analysis**: Multi-Model MobileNetV1 1D Comparison
**Models Analyzed**: {len(model_names)}
**Data**: ECG Classification (5 classes, {sum(models_info[name].get('total_params', 0) for name in model_names if name in models_info):,} total parameters analyzed)
"""

        # Save report
        with open(f'{output_dir}/three_way_comparison_report.md', 'w') as f:
            f.write(report_content)
        
        # Save deployment recommendations as JSON
        deployment_summary = {
            'overall_winner': model_scores[0][0],
            'overall_score': float(model_scores[0][1]),
            'mobile_recommendation': mobile_best[0] if mobile_best else "None meet requirements",
            'clinical_recommendation': model_scores[0][0],
            'rankings': [
                {
                    'rank': i,
                    'model': name,
                    'score': float(score),
                    'test_accuracy': float(result['test_accuracy']),
                    'generalization_gap': float(info['generalization_gap']),
                    'model_size_mb': float(info['model_size_mb']),
                    'inference_time_ms': float(result['avg_inference_time'])
                }
                for i, (name, score, info, result) in enumerate(model_scores, 1)
            ],
            'gap_compliance': {
                name: {
                    'generalization_compliant': bool(models_info[name]['generalization_gap'] < 5),
                    'accuracy_gap_compliant': bool(2 <= models_info[name]['accuracy_gap'] <= 5),
                    'f1_gap_compliant': bool(1 <= models_info[name]['f1_gap'] <= 4),
                    'fully_compliant': bool(all([
                        models_info[name]['generalization_gap'] < 5,
                        2 <= models_info[name]['accuracy_gap'] <= 5,
                        1 <= models_info[name]['f1_gap'] <= 4
                    ]))
                }
                for name in model_names if name in models_info
            },
            'generated_at': datetime.now().isoformat()
        }
        
        with open(f'{output_dir}/three_way_deployment_summary.json', 'w') as f:
            json.dump(deployment_summary, f, indent=2)
        
        return report_content


def main():
    """Main function to generate 3-way comparison report"""
    print("MOBILENETV1 1D ECG - MODEL COMPARISON GENERATOR")
    print("="*80)
    
    # Initialize reporter
    reporter = ModelComparisonReporter()
    
    # Define all possible model paths
    model_configs = [
        {
            'name': 'Approach 1 (Performance-First)',
            'paths': [
                "outputs/models/approach1_performance_first/mobilenet_best.pth",
                "outputs/models/approach1/mobilenet_best.pth",
                "outputs/models/performance_first/mobilenet_best.pth"
            ]
        },
        {
            'name': 'Approach 2 - Current Best',
            'paths': [
                "outputs/models/strategy2_tight_gaps/mobilenet_best.pth",
                "outputs/models/approach2/mobilenet_best.pth",
                "outputs/models/generalization_first/mobilenet_best.pth"
            ]
        },
        {
            'name': 'Approach 2 - Backup (Epoch 32)',
            'paths': [
                "outputs/models/strategy2_tight_gaps/mobilenet_epoch38_backup.pth",
                "outputs/models/strategy2_tight_gaps/mobilenet_backup.pth",
                "outputs/models/approach2_backup/mobilenet_best.pth"
            ]