import onnx
from onnx_tf.backend import prepare
import os

# Paths
ONNX_MODEL_PATH = 'assets/ml_models/mobilenet_v1_ecg_optimized_90plus.onnx'  # Optimized 90plus ONNX model
TF_SAVED_MODEL_DIR = 'tf_saved_model'  # Output directory for TensorFlow SavedModel

# Step 1: Convert ONNX to TensorFlow SavedModel
print('🔄 Loading ONNX model...')
onnx_model = onnx.load(ONNX_MODEL_PATH)
print('✅ ONNX model loaded.')

print('🔄 Converting to TensorFlow SavedModel...')
tf_rep = prepare(onnx_model)
os.makedirs(TF_SAVED_MODEL_DIR, exist_ok=True)
tf_rep.export_graph(TF_SAVED_MODEL_DIR)
print(f'✅ TensorFlow SavedModel exported to {TF_SAVED_MODEL_DIR}/')

print('🎉 ONNX to TensorFlow conversion complete!')
print('Next step: Run the following command in your terminal:')
print('tensorflowjs_converter --input_format=tf_saved_model tf_saved_model tfjs_model/')
