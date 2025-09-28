"""
GPU Detection and Installation Helper
Run this to check GPU status and install GPU PyTorch if needed
"""

import subprocess
import sys

def check_gpu_status():
    """Check current GPU status and PyTorch installation"""
    
    print("🔍 GPU DETECTION AND SETUP")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"🚀 CUDA available: YES")
            print(f"   CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') and torch.version.cuda else 'N/A'}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"         Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print(f"❌ CUDA available: NO")
            print(f"   This means PyTorch is CPU-only version")
            
    except ImportError:
        print(f"❌ PyTorch not installed")
    
    # Check NVIDIA GPU hardware
    print(f"\n🖥️  HARDWARE CHECK:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ NVIDIA GPU detected")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    print(f"   {line.strip()}")
        else:
            print(f"❌ nvidia-smi not available")
    except:
        print(f"❌ nvidia-smi command failed")

def install_gpu_pytorch():
    """Install GPU version of PyTorch"""
    
    print(f"\n🔧 INSTALLING GPU PYTORCH")
    print("=" * 50)
    
    commands = [
        # Uninstall CPU version
        "conda uninstall pytorch torchvision torchaudio cpuonly -y",
        # Install GPU version  
        "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y"
    ]
    
    for cmd in commands:
        print(f"🔄 Running: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Success")
            else:
                print(f"❌ Failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_gpu_training():
    """Test GPU training with a small example"""
    
    print(f"\n🧪 TESTING GPU TRAINING")
    print("=" * 50)
    
    try:
        import torch
        import torch.nn as nn
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Testing on GPU: {torch.cuda.get_device_name(0)}")
            
            # Simple test
            x = torch.randn(100, 10).to(device)
            model = nn.Linear(10, 5).to(device)
            y = model(x)
            
            print(f"✅ GPU training test successful!")
            print(f"   Input shape: {x.shape}")
            print(f"   Output shape: {y.shape}")
            print(f"   Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            
        else:
            print(f"❌ No GPU available for testing")
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")

if __name__ == "__main__":
    check_gpu_status()
    
    # Ask user if they want to install GPU PyTorch
    try:
        import torch
        if not torch.cuda.is_available():
            response = input("\n🤔 Install GPU PyTorch? (y/N): ")
            if response.lower() == 'y':
                install_gpu_pytorch()
                print(f"\n🔄 Please restart your script after installation completes")
        else:
            test_gpu_training()
    except ImportError:
        print(f"\n🚨 PyTorch not found. Please install PyTorch first.")
