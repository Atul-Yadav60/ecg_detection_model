#!/usr/bin/env python3
"""
Environment Validation Script for ECG Arrhythmia Detection
Optimized for RTX 3050 setup
"""

import sys
import torch
import numpy as np
import pandas as pd
import subprocess
import platform
import json
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Python Environment Check:")
    print(f"   Python Version: {sys.version}")
    print(f"   Platform: {platform.platform()}")
    
    if sys.version_info >= (3, 9):
        print("   ✅ Python version OK (3.9+)")
    else:
        print("   ❌ Python version too old, need 3.9+")
        return False
    return True

def check_cuda_installation():
    """Check CUDA installation and GPU"""
    print("\n🚀 CUDA & GPU Check:")
    
    try:
        # Check nvidia-smi
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ nvidia-smi working")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX 3050' in line:
                    print(f"   🎮 Found: RTX 3050 Laptop GPU")
                    break
        else:
            print("   ❌ nvidia-smi not working")
            return False
            
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found")
        return False
    
    return True

def check_pytorch_gpu():
    """Check PyTorch GPU support"""
    print("\n🔥 PyTorch GPU Check:")
    print(f"   PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("   ✅ CUDA available in PyTorch")
        print(f"   🎮 GPU Count: {torch.cuda.device_count()}")
        
        # Get GPU info
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   📊 GPU {i}: {gpu_name}")
            print(f"   💾 VRAM: {gpu_memory:.1f} GB")
            
            # Test GPU computation
            try:
                test_tensor = torch.randn(1000, 1000).cuda(i)
                result = torch.mm(test_tensor, test_tensor.t())
                print(f"   ✅ GPU {i} computation test passed")
                
                # Memory test for 4GB VRAM
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / 1e9
                cached = torch.cuda.memory_reserved(i) / 1e9
                print(f"   📈 Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
                
            except Exception as e:
                print(f"   ❌ GPU {i} computation test failed: {e}")
                return False
                
        return True
    else:
        print("   ❌ CUDA not available in PyTorch")
        return False

def check_required_packages():
    """Check if all required packages are installed"""
    print("\n📦 Package Installation Check:")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'wfdb': 'WFDB (PhysioNet)',
        'neurokit2': 'NeuroKit2',
        'biosppy': 'BioSPPy',
        'streamlit': 'Streamlit',
        'tensorboard': 'TensorBoard',
        'tqdm': 'TQDM',
        'plotly': 'Plotly'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   📝 Install missing packages with:")
        for package in missing_packages:
            print(f"      pip install {package}")
        return len(missing_packages) == 0  # Return False if any missing
    
    return True

def test_gpu_memory_optimization():
    """Test GPU memory settings for RTX 3050"""
    print("\n🧠 GPU Memory Optimization Test:")
    
    if not torch.cuda.is_available():
        print("   ⚠️  Skipping - CUDA not available")
        return True
    
    try:
        # Test different batch sizes to find optimal
        device = torch.device('cuda:0')
        
        # Simulate ECG data (single lead, 250Hz, 10 seconds)
        sequence_length = 2500
        
        optimal_batch_sizes = []
        
        for batch_size in [8, 16, 32, 64, 128]:
            try:
                torch.cuda.empty_cache()
                
                # Create dummy ECG batch
                dummy_data = torch.randn(batch_size, 1, sequence_length).to(device)
                
                # Simple CNN forward pass
                conv1 = torch.nn.Conv1d(1, 32, 15, padding=7).to(device)
                conv2 = torch.nn.Conv1d(32, 64, 15, padding=7).to(device)
                pool = torch.nn.AdaptiveAvgPool1d(512).to(device)
                
                with torch.no_grad():
                    x = conv1(dummy_data)
                    x = torch.relu(x)
                    x = conv2(x)
                    x = torch.relu(x)
                    x = pool(x)
                
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f"   ✅ Batch size {batch_size:3d}: {memory_used:.2f}GB VRAM")
                
                if memory_used < 3.5:  # Safe margin for 4GB
                    optimal_batch_sizes.append(batch_size)
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   ❌ Batch size {batch_size:3d}: OOM")
                    break
                else:
                    raise e
        
        if optimal_batch_sizes:
            max_batch = max(optimal_batch_sizes)
            print(f"   🎯 Recommended max batch size: {max_batch}")
            print(f"   💡 Safe batch sizes: {optimal_batch_sizes}")
        else:
            print("   ⚠️  No suitable batch size found - check GPU memory")
            
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"   ❌ GPU memory test failed: {e}")
        return False

def test_mixed_precision():
    """Test mixed precision training support"""
    print("\n⚡ Mixed Precision Training Test:")
    
    if not torch.cuda.is_available():
        print("   ⚠️  Skipping - CUDA not available")
        return True
    
    try:
        from torch.cuda.amp import GradScaler, autocast
        
        device = torch.device('cuda:0')
        model = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, 15, padding=7),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 15, padding=7),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(128),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 128, 5)  # 5 arrhythmia classes
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler()
        criterion = torch.nn.CrossEntropyLoss()
        
        # Test data
        dummy_data = torch.randn(16, 1, 2500).to(device)
        dummy_labels = torch.randint(0, 5, (16,)).to(device)
        
        # Test mixed precision forward pass
        with autocast():
            outputs = model(dummy_data)
            loss = criterion(outputs, dummy_labels)
        
        # Test backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("   ✅ Mixed precision training supported")
        print("   💡 This will reduce VRAM usage significantly")
        
        # Test memory usage
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f"   📊 Memory usage with mixed precision: {memory_used:.2f}GB")
        
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"   ❌ Mixed precision test failed: {e}")
        return False

def create_environment_summary():
    """Create environment summary file"""
    print("\n📋 Creating Environment Summary...")
    
    summary = {
        'timestamp': str(pd.Timestamp.now()),
        'python_version': sys.version,
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        summary['gpu_name'] = torch.cuda.get_device_name(0)
        summary['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        summary['cuda_version'] = torch.version.cuda
    
    # Test optimal batch size
    if torch.cuda.is_available():
        optimal_batches = []
        for batch_size in [16, 32, 64]:
            try:
                torch.cuda.empty_cache()
                dummy = torch.randn(batch_size, 1, 2500).cuda()
                conv = torch.nn.Conv1d(1, 32, 15).cuda()
                _ = conv(dummy)
                memory_used = torch.cuda.memory_allocated() / 1e9
                if memory_used < 3.5:
                    optimal_batches.append(batch_size)
                torch.cuda.empty_cache()
            except:
                break
        summary['recommended_batch_sizes'] = optimal_batches
    
    # Save summary
    with open('environment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("   ✅ Environment summary saved to 'environment_summary.json'")
    return summary

def main():
    """Main validation function"""
    print("=" * 70)
    print("🏥 ECG Arrhythmia Detection - Environment Validation")
    print("💻 ASUS TUF A15 with RTX 3050 - Optimized Setup")
    print("=" * 70)
    
    all_checks = [
        check_python_version(),
        check_cuda_installation(),
        check_pytorch_gpu(),
        check_required_packages(),
        test_gpu_memory_optimization(),
        test_mixed_precision()
    ]
    
    print("\n" + "=" * 70)
    if all(all_checks):
        print("🎉 ALL CHECKS PASSED! Environment ready for ECG training")
        print("\n🚀 RTX 3050 Optimized Settings:")
        print("   • Recommended batch size: 32-64 (training), 1-8 (inference)")
        print("   • Use mixed precision training (AMP) - ENABLED ✅")
        print("   • Target memory usage: <3.5GB VRAM")
        print("   • Expected training time: 2-4 hours for full dataset")
        print("   • Real-time inference: <100ms latency")
    else:
        failed_checks = sum(1 for check in all_checks if not check)
        print(f"❌ {failed_checks} checks failed. Please fix issues above.")
        print("\n💡 Common solutions:")
        print("   • Install missing packages: pip install <package_name>")
        print("   • Restart terminal after package installation")
        print("   • Update conda: conda update conda")
    
    # Create summary regardless
    summary = create_environment_summary()
    
    if all(all_checks):
        print(f"\n📊 System Summary:")
        print(f"   • PyTorch: {summary['torch_version']}")
        print(f"   • GPU: {summary.get('gpu_name', 'N/A')}")
        print(f"   • VRAM: {summary.get('gpu_memory_gb', 0):.1f}GB")
        print(f"   • Optimal batch sizes: {summary.get('recommended_batch_sizes', [])}")
    
    print("=" * 70)
    
    return all(all_checks)

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Ready for Step 2: Project Structure Creation!")
    else:
        print("\n🔧 Please resolve the issues above before proceeding.")