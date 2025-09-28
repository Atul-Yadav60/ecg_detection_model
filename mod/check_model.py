# save as check_model.py
import torch
import sys

try:
    checkpoint = torch.load('outputs/models/strategy2_tight_gaps/mobilenet_best.pth', map_location='cpu')
    
    epoch = checkpoint.get('best_metrics', {}).get('epoch', 'Unknown')
    val_acc = checkpoint.get('best_metrics', {}).get('val_acc', 0)
    val_f1 = checkpoint.get('best_metrics', {}).get('val_f1', 0)
    gen_gap = checkpoint.get('best_metrics', {}).get('generalization_gap', 0)
    
    print(f"🏆 BEST MODEL INFO:")
    print(f"   Saved from Epoch: {epoch}")
    print(f"   Validation Accuracy: {val_acc:.2f}%")
    print(f"   Validation F1: {val_f1:.2f}%")
    print(f"   Generalization Gap: {gen_gap:.2f}%")
    
    if 'history' in checkpoint and checkpoint['history']['epochs']:
        total_epochs = len(checkpoint['history']['epochs'])
        last_epoch = max(checkpoint['history']['epochs'])
        print(f"   Total epochs completed: {total_epochs}")
        print(f"   Training stopped at epoch: {last_epoch}")
    
except Exception as e:
    print(f"Error reading checkpoint: {e}")