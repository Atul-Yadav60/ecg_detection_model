import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx2keras import onnx_to_keras
import tensorflowjs as tfjs
import os

# ---- USER CONFIG ----
PYTORCH_MODEL_PATH = '../best_model_optimized_90plus.pth'  # Path to your trained PyTorch model
MODEL_CLASS_NAME = 'MobileNetV1_1D'  # Change to your actual class name if different
INPUT_SHAPE = (1000, 1)  # Match your ONNX and training config
OUTPUT_CLASSES = 5        # Number of output classes
DUMMY_BATCH_SIZE = 1      # For export
ONNX_EXPORT_PATH = 'temp_model.onnx'
KERAS_EXPORT_PATH = 'ecg_model_keras.h5'
TFJS_EXPORT_DIR = 'tfjs_model'

# ---- MODEL DEFINITION ----
# Import your model class
from FINAL_CORRECT_MODEL import MobileNetV1_1D

# ---- STEP 1: Load PyTorch Model ----
print('🔄 Loading PyTorch model...')
model = MobileNetV1_1D(num_classes=OUTPUT_CLASSES)
state_dict = torch.load(PYTORCH_MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
print('✅ PyTorch model loaded.')

# ---- STEP 2: Export to ONNX ----
print('🔄 Exporting to ONNX...')
dummy_input = torch.randn(DUMMY_BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1])
torch.onnx.export(model, dummy_input, ONNX_EXPORT_PATH,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                  opset_version=11)
print(f'✅ ONNX model exported to {ONNX_EXPORT_PATH}')

# ---- STEP 3: Convert ONNX to Keras ----
print('🔄 Converting ONNX to Keras...')
onnx_model = onnx.load(ONNX_EXPORT_PATH)
k_model = onnx_to_keras(onnx_model, ['input'], name_policy='short')
k_model.save(KERAS_EXPORT_PATH)
print(f'✅ Keras model saved to {KERAS_EXPORT_PATH}')

# ---- STEP 4: Convert Keras to TensorFlow.js ----
print('🔄 Converting Keras to TensorFlow.js...')
os.makedirs(TFJS_EXPORT_DIR, exist_ok=True)
tfjs.converters.save_keras_model(k_model, TFJS_EXPORT_DIR)
print(f'✅ TensorFlow.js model saved to {TFJS_EXPORT_DIR}/model.json')

print('🎉 All conversions completed successfully!')
