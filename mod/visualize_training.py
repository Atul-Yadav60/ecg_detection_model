#!/usr/bin/env python3
"""
Visualize ECG training results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_training_history(model_name='simple'):
    """Load training history from JSON file"""
    history_file = f'outputs/models/{model_name}_training_history.json'
    
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    return data

def plot_training_curves(history_data):
    """Plot training and validation curves"""
    
    history = history_data['history']
    epochs = history['epochs']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    ax3.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Best epoch indicator
    best_epoch = history_data['best_epoch']
    ax4.axvline(x=best_epoch, color='r', linestyle='--', linewidth=2, 
                label=f'Best Epoch: {best_epoch}')
    ax4.plot(epochs, history['val_acc'], 'b-', linewidth=2, label='Validation Accuracy')
    ax4.set_title(f'Best Model at Epoch {best_epoch}', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    model_name = history_data['model_name'].upper()
    best_acc = history_data['best_val_acc']
    plt.suptitle(f'Training Summary - {model_name} Model\nBest Validation Accuracy: {best_acc:.2f}%', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('outputs/reports')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f'{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Training curves saved to outputs/reports/{model_name}_training_curves.png")

def print_training_summary(history_data):
    """Print training summary"""
    
    print(f"\n{'='*60}")
    print(f"📊 TRAINING SUMMARY - {history_data['model_name'].upper()}")
    print(f"{'='*60}")
    
    config = history_data['config']
    history = history_data['history']
    
    print(f"🏗️  Model Configuration:")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Weight Decay: {config['weight_decay']}")
    
    print(f"\n📈 Training Results:")
    print(f"   Best Validation Accuracy: {history_data['best_val_acc']:.2f}%")
    print(f"   Best Epoch: {history_data['best_epoch']}")
    print(f"   Total Training Time: {len(history['epochs'])} epochs")
    
    print(f"\n📊 Final Metrics:")
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    print(f"   Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"   Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"   Final Training Loss: {final_train_loss:.4f}")
    print(f"   Final Validation Loss: {final_val_loss:.4f}")
    
    # Check for overfitting
    acc_gap = final_train_acc - final_val_acc
    loss_gap = final_val_loss - final_train_loss
    
    print(f"\n🔍 Overfitting Analysis:")
    print(f"   Accuracy Gap (Train - Val): {acc_gap:.2f}%")
    print(f"   Loss Gap (Val - Train): {loss_gap:.4f}")
    
    if acc_gap > 5:
        print(f"   ⚠️  Potential overfitting detected (accuracy gap > 5%)")
    else:
        print(f"   ✅ Good generalization (accuracy gap <= 5%)")

def main():
    """Main function"""
    
    print("📊 ECG Training Visualization")
    print("="*40)
    
    try:
        # Load training history
        history_data = load_training_history('simple')
        
        # Print summary
        print_training_summary(history_data)
        
        # Plot curves
        plot_training_curves(history_data)
        
        print(f"\n✅ Visualization completed!")
        print(f"📁 Check outputs/reports/ for generated plots")
        
    except FileNotFoundError:
        print("❌ Training history file not found!")
        print("Run training first: python train_ecg.py simple --quick-test")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
