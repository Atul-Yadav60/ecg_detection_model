#!/usr/bin/env python3
"""
Compare ECG model training results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_all_training_histories():
    """Load training histories for all models"""
    models = ['simple', 'cnn_lstm']
    histories = {}
    
    for model in models:
        try:
            history_file = f'outputs/models/{model}_training_history.json'
            with open(history_file, 'r') as f:
                histories[model] = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  No training history found for {model}")
    
    return histories

def plot_model_comparison(histories):
    """Plot comparison of all models"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'simple': 'blue', 'cnn_lstm': 'red'}
    markers = {'simple': 'o', 'cnn_lstm': 's'}
    
    # Accuracy comparison
    for model_name, history_data in histories.items():
        epochs = history_data['history']['epochs']
        val_acc = history_data['history']['val_acc']
        train_acc = history_data['history']['train_acc']
        
        ax1.plot(epochs, val_acc, color=colors[model_name], marker=markers[model_name], 
                linewidth=2, label=f'{model_name.upper()} (Val)', alpha=0.8)
        ax1.plot(epochs, train_acc, color=colors[model_name], marker=markers[model_name], 
                linewidth=2, linestyle='--', label=f'{model_name.upper()} (Train)', alpha=0.6)
    
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(75, 95)
    
    # Loss comparison
    for model_name, history_data in histories.items():
        epochs = history_data['history']['epochs']
        val_loss = history_data['history']['val_loss']
        train_loss = history_data['history']['train_loss']
        
        ax2.plot(epochs, val_loss, color=colors[model_name], marker=markers[model_name], 
                linewidth=2, label=f'{model_name.upper()} (Val)', alpha=0.8)
        ax2.plot(epochs, train_loss, color=colors[model_name], marker=markers[model_name], 
                linewidth=2, linestyle='--', label=f'{model_name.upper()} (Train)', alpha=0.6)
    
    ax2.set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Best performance comparison
    model_names = []
    best_accs = []
    best_epochs = []
    param_counts = []
    
    for model_name, history_data in histories.items():
        model_names.append(model_name.upper())
        best_accs.append(history_data['best_val_acc'])
        best_epochs.append(history_data['best_epoch'])
        
        # Get parameter count from model info
        if model_name == 'simple':
            param_counts.append(195909)
        elif model_name == 'cnn_lstm':
            param_counts.append(569861)
    
    # Bar chart of best accuracies
    bars = ax3.bar(model_names, best_accs, color=[colors[m.lower()] for m in model_names], alpha=0.7)
    ax3.set_title('Best Validation Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_ylim(80, 90)
    
    # Add value labels on bars
    for bar, acc in zip(bars, best_accs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Parameter count comparison
    bars2 = ax4.bar(model_names, param_counts, color=[colors[m.lower()] for m in model_names], alpha=0.7)
    ax4.set_title('Model Complexity (Parameters)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Parameters')
    
    # Add value labels on bars
    for bar, params in zip(bars2, param_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 10000,
                f'{params:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('ECG Model Comparison - Quick Test Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('outputs/reports')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Model comparison saved to outputs/reports/model_comparison.png")

def print_benchmark_summary(histories):
    """Print comprehensive benchmark summary"""
    
    print(f"\n{'='*80}")
    print(f"🏆 COMPREHENSIVE BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"\n{'Model':<12} {'Best Acc':<10} {'Best Epoch':<12} {'Parameters':<12} {'Overfitting':<12}")
    print("-" * 70)
    
    for model_name, history_data in histories.items():
        best_acc = history_data['best_val_acc']
        best_epoch = history_data['best_epoch']
        
        # Calculate overfitting
        final_train_acc = history_data['history']['train_acc'][-1]
        final_val_acc = history_data['history']['val_acc'][-1]
        overfitting_gap = final_train_acc - final_val_acc
        
        # Get parameter count
        if model_name == 'simple':
            params = 195909
        elif model_name == 'cnn_lstm':
            params = 569861
        
        overfitting_status = "High" if overfitting_gap > 5 else "Low"
        
        print(f"{model_name.upper():<12} {best_acc:<10.2f}% {best_epoch:<12} {params:<12,} {overfitting_status:<12}")
    
    # Find winner
    best_model = max(histories.keys(), key=lambda x: histories[x]['best_val_acc'])
    best_acc = histories[best_model]['best_val_acc']
    
    print(f"\n🏆 WINNER: {best_model.upper()} with {best_acc:.2f}% validation accuracy")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Run full training with early stopping for {best_model}")
    print(f"   2. Try ResNet1D and Transformer models for potentially better performance")
    print(f"   3. Address overfitting with regularization techniques")
    print(f"   4. Consider class imbalance (N class dominates at 79.3%)")

def main():
    """Main function"""
    
    print("🏆 ECG Model Comparison")
    print("="*40)
    
    try:
        # Load all training histories
        histories = load_all_training_histories()
        
        if not histories:
            print("❌ No training histories found!")
            print("Run training first: python train_ecg.py --benchmark --quick-test")
            return
        
        # Print summary
        print_benchmark_summary(histories)
        
        # Plot comparison
        plot_model_comparison(histories)
        
        print(f"\n✅ Comparison completed!")
        print(f"📁 Check outputs/reports/ for generated plots")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
