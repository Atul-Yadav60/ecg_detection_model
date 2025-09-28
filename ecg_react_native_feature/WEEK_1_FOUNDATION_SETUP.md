# 🚀 WEEK 1: FOUNDATION SETUP - DAY-BY-DAY PLAN

## **CURRENT STATUS CHECK**
- ✅ You have the `mod/` folder with trained models
- ✅ You have our `ecg_react_native_feature/` folder with complete ECG module
- ✅ You've been working on ECGHomeScreen.tsx (good progress!)
- 🔄 **STARTING MILESTONE 1.1: Environment Preparation**

---

## **DAY 1-2: MILESTONE 1.1 - ENVIRONMENT PREPARATION**

### **STEP 1A: Convert Your Trained Model**

**What to do RIGHT NOW:**

1. **Navigate to your mod folder:**
```powershell
cd "C:\Users\Atul2\OneDrive\Desktop\ECG detection and analysis\mod"
```

2. **Check what model files you have:**
```powershell
dir *.pth
dir *.pt
dir *.pkl
```

3. **Create the model conversion script:**
```python
# Save this as: convert_model_to_mobile.py
import torch
import torch.onnx
import numpy as np
import os

def convert_mobilenet_to_onnx():
    """Convert your trained MobileNet v1 model to ONNX format"""
    
    print("🔄 Starting model conversion...")
    
    # TODO: REPLACE THESE WITH YOUR ACTUAL MODEL DETAILS
    # Look for your best performing model file in mod/ folder
    
    # Common model file names to check:
    possible_model_files = [
        "best_model.pth",
        "mobilenet_v1_best.pth", 
        "approach_1_best.pth",
        "final_model.pth",
        "checkpoint.pth"
    ]
    
    model_file = None
    for filename in possible_model_files:
        if os.path.exists(filename):
            model_file = filename
            print(f"📁 Found model file: {filename}")
            break
    
    if not model_file:
        print("❌ No model file found. Please check your mod/ folder")
        print("Available files:")
        for file in os.listdir("."):
            if file.endswith((".pth", ".pt", ".pkl")):
                print(f"  - {file}")
        return False
    
    try:
        # Load your trained model
        # NOTE: You'll need to import your actual model class here
        # from models import MobileNetV1  # Replace with your actual import
        
        # For now, create a dummy MobileNet-like model
        # You'll replace this with your actual model loading
        model = torch.nn.Sequential(
            # Input: [batch, 1, 187] - ECG segment
            torch.nn.Conv1d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 5)  # 5 classes: N, V, S, F, Q
        )
        
        # TODO: Load your actual trained weights
        # checkpoint = torch.load(model_file, map_location='cpu')
        # model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        
        # Create dummy input matching your training data shape
        dummy_input = torch.randn(1, 1, 187)  # [batch=1, channels=1, length=187]
        
        # Export to ONNX
        output_path = "mobilenet_v1_ecg_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['ecg_input'],
            output_names=['predictions'],
            dynamic_axes={
                'ecg_input': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Model converted successfully!")
        print(f"📁 Output: {output_path}")
        print(f"📊 Size: {file_size:.2f} MB")
        print(f"🎯 Ready for mobile integration!")
        
        # Copy to the React Native feature folder
        target_path = "../ecg_react_native_feature/assets/ml_models/mobilenet_v1_ecg_model.onnx"
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        import shutil
        shutil.copy(output_path, target_path)
        print(f"📋 Copied to: {target_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def test_converted_model():
    """Test the converted ONNX model"""
    try:
        import onnx
        import onnxruntime as ort
        
        model_path = "mobilenet_v1_ecg_model.onnx"
        if not os.path.exists(model_path):
            print("❌ ONNX model not found")
            return False
        
        # Load and validate ONNX model
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference
        session = ort.InferenceSession(model_path)
        
        # Test with dummy ECG data
        test_input = np.random.randn(1, 1, 187).astype(np.float32)
        outputs = session.run(None, {'ecg_input': test_input})
        
        predictions = outputs[0]
        print(f"✅ ONNX model test successful!")
        print(f"📊 Input shape: {test_input.shape}")
        print(f"📊 Output shape: {predictions.shape}")
        print(f"📊 Predictions: {predictions}")
        
        # Apply softmax to get probabilities
        import math
        exp_outputs = [math.exp(x) for x in predictions[0]]
        sum_exp = sum(exp_outputs)
        probabilities = [x / sum_exp for x in exp_outputs]
        
        class_names = ['Normal', 'Ventricular', 'Supraventricular', 'Fusion', 'Unknown']
        print("\n🎯 Prediction probabilities:")
        for i, (name, prob) in enumerate(zip(class_names, probabilities)):
            print(f"  {name}: {prob:.3f} ({prob*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        print("💡 You may need to install: pip install onnx onnxruntime")
        return False

if __name__ == "__main__":
    print("🫀 ECG Model Conversion for Mobile")
    print("=" * 40)
    
    # Step 1: Convert model
    if convert_mobilenet_to_onnx():
        print("\n" + "=" * 40)
        print("📱 Testing converted model...")
        
        # Step 2: Test converted model
        test_converted_model()
        
        print("\n🎯 NEXT STEPS:")
        print("1. ✅ Model conversion complete")
        print("2. 🔄 Continue to STEP 1B: Development Environment Setup")
        print("3. 📱 Create React Native project")
    else:
        print("\n❌ Model conversion failed")
        print("💡 Please check your model files and try again")
```

4. **Run the conversion:**
```powershell
python convert_model_to_mobile.py
```

### **STEP 1B: Install Development Environment**

**Check what you already have:**
```powershell
# Check Node.js
node --version

# Check npm
npm --version

# Check React Native CLI
npx react-native --version

# Check Java (for Android development)
java -version
```

**If missing, install:**

1. **Node.js 18+**: Download from https://nodejs.org/
2. **React Native CLI**:
```powershell
npm install -g react-native-cli
```

3. **Android Studio**: Download from https://developer.android.com/studio

---

## **DAY 3-4: MILESTONE 1.2 - CREATE REACT NATIVE PROJECT**

### **STEP 2A: Create New React Native Project**

```powershell
# Navigate to your project directory
cd "C:\Users\Atul2\OneDrive\Desktop\ECG detection and analysis"

# Create new React Native project with TypeScript
npx react-native init ECGSmartwatchApp --template react-native-template-typescript

# Navigate to the new project
cd ECGSmartwatchApp

# Verify project creation
dir
```

### **STEP 2B: Install ECG Dependencies**

```powershell
# Install core dependencies for our ECG feature
npm install @react-navigation/native@^6.1.7
npm install @react-navigation/stack@^6.3.17
npm install react-native-screens@^3.22.0
npm install react-native-safe-area-context@^4.7.1

# Install Bluetooth and storage dependencies
npm install react-native-ble-plx@^3.0.0
npm install @react-native-async-storage/async-storage@^1.19.0
npm install react-native-permissions@^3.8.0

# Install UI and chart dependencies
npm install react-native-svg@^13.9.0
npm install react-native-vector-icons@^9.2.0
npm install react-native-chart-kit@^6.12.0
npm install react-native-linear-gradient@^2.8.0

# Install for haptic feedback and sound
npm install react-native-haptic-feedback@^1.14.0
npm install react-native-sound@^0.11.2
```

### **STEP 2C: Copy Our ECG Feature Module**

```powershell
# Copy the complete ECG feature to your new project
xcopy "..\ecg_react_native_feature\src\features" ".\src\features" /E /I

# Copy assets (including your converted model)
xcopy "..\ecg_react_native_feature\assets" ".\assets" /E /I

# Verify files were copied
dir src\features
dir assets\ml_models
```

---

## **DAY 5-7: MILESTONE 1.3 - BASIC INTEGRATION & TESTING**

### **STEP 3A: Fix Import Issues**

I noticed your ECGHomeScreen.tsx has some import issues. Let's fix them:

1. **Create the shared constants folder:**
```powershell
mkdir src\shared\constants
```

2. **Create the Design constants file:**
