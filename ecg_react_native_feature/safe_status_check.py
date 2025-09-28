"""
SAFE Training Status Checker - READ ONLY
This script only reads files and doesn't interfere with running training
"""

import os
import time
from datetime import datetime

def safe_training_check():
    """Safely check training status without interference"""
    
    print("🔍 SAFE TRAINING STATUS CHECK (READ-ONLY)")
    print("=" * 55)
    print("⚠️  This script ONLY reads files - no interference with training")
    
    # Check for model files (indicates training progress)
    model_files = []
    base_dir = "C:\\Users\\Atul2\\OneDrive\\Desktop\\ECG detection and analysis"
    
    print(f"\n📂 Checking directory: {base_dir}")
    
    for i in range(5):  # Check for up to 5 folds
        model_path = os.path.join(base_dir, f"best_model_fold_{i}.pth")
        if os.path.exists(model_path):
            stat = os.stat(model_path)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            time_diff = datetime.now() - mod_time
            
            model_files.append({
                'file': f"best_model_fold_{i}.pth",
                'size_mb': size_mb,
                'modified': mod_time,
                'age_minutes': time_diff.total_seconds() / 60
            })
    
    if model_files:
        print(f"\n✅ FOUND {len(model_files)} MODEL CHECKPOINT(S):")
        for model in model_files:
            age_str = f"{model['age_minutes']:.1f} minutes ago"
            if model['age_minutes'] < 5:
                status = "🔥 VERY RECENT (likely active training)"
            elif model['age_minutes'] < 30:
                status = "🟡 RECENT (training may be ongoing)"
            else:
                status = "⚪ OLDER (training may be paused/completed)"
            
            print(f"   📄 {model['file']}")
            print(f"      Size: {model['size_mb']:.1f} MB")
            print(f"      Modified: {model['modified']}")
            print(f"      Age: {age_str}")
            print(f"      Status: {status}")
            print()
    else:
        print(f"\n❌ NO MODEL CHECKPOINTS FOUND")
        print(f"   Training may not have started or files are in different location")
    
    # Check for any Python processes (non-intrusive)
    print(f"\n🔒 SAFETY GUARANTEE:")
    print(f"   ✅ This script only READ files")
    print(f"   ✅ No training processes were touched")
    print(f"   ✅ No files were modified")
    print(f"   ✅ Safe to run anytime")
    
    # Check for log files or other indicators
    log_patterns = ["*.log", "training_*.txt", "*.out"]
    print(f"\n📋 CHECKING FOR LOG FILES:")
    
    found_logs = False
    for pattern in ["training_log.txt", "output.log", "training_output.txt"]:
        log_path = os.path.join(base_dir, pattern)
        if os.path.exists(log_path):
            stat = os.stat(log_path)
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            print(f"   📝 Found: {pattern} (modified: {mod_time})")
            found_logs = True
    
    if not found_logs:
        print(f"   ❌ No standard log files found")
    
    # Final assessment
    print(f"\n🎯 TRAINING ASSESSMENT:")
    print("=" * 55)
    
    if model_files:
        latest_model = max(model_files, key=lambda x: x['modified'])
        if latest_model['age_minutes'] < 10:
            print(f"🔥 LIKELY ACTIVE: Latest model is very recent ({latest_model['age_minutes']:.1f} min ago)")
            print(f"   Recommendation: DO NOT run new training commands")
        elif latest_model['age_minutes'] < 60:
            print(f"🟡 POSSIBLY ACTIVE: Recent activity ({latest_model['age_minutes']:.1f} min ago)")
            print(f"   Recommendation: Check task manager for Python processes")
        else:
            print(f"⚪ LIKELY STOPPED: No recent activity ({latest_model['age_minutes']:.1f} min ago)")
            print(f"   Recommendation: Safe to restart training if needed")
    else:
        print(f"❓ UNKNOWN: No training evidence found")
        print(f"   Recommendation: Safe to start training")

if __name__ == "__main__":
    safe_training_check()
