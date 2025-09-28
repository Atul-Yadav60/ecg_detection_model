import onnxruntime as ort
import numpy as np
import tensorflow as tf
import json
import os

def extract_onnx_weights(onnx_path):
    """Extract weights from ONNX model"""
    print("🔄 Loading ONNX model with ONNX Runtime...")

    # Create ONNX session
    session = ort.InferenceSession(onnx_path)

    # Get model metadata
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"Input: {inputs[0].name}, shape: {inputs[0].shape}")
    print(f"Output: {outputs[0].name}, shape: {outputs[0].shape}")

    # Get all initializers (weights)
    weights = {}
    for initializer in session.get_modelmeta().custom_metadata_map:
        print(f"Found metadata: {initializer}")

    # Since we can't easily extract weights from ONNX Runtime,
    # let's use a different approach - convert using onnx-tf
    return session

def create_tfjs_model_from_onnx(onnx_path, output_dir):
    """Create TensorFlow.js model by converting ONNX to TF then to TFJS"""

    print("🔄 Converting ONNX to TensorFlow SavedModel...")

    # Use onnx-tf to convert ONNX to TensorFlow
    import onnx_tf
    from onnx_tf.backend import prepare

    import onnx
    onnx_model = onnx.load(onnx_path)

    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)
    tf_model_path = os.path.join(output_dir, 'tf_model')

    # Save as TensorFlow SavedModel
    tf_rep.export_graph(tf_model_path)
    print(f"✅ TensorFlow model saved to: {tf_model_path}")

    # Now convert TensorFlow to TensorFlow.js manually
    print("🔄 Converting to TensorFlow.js format...")

    # Load the TensorFlow model
    imported = tf.saved_model.load(tf_model_path)

    # Get the concrete function
    concrete_func = imported.signatures['serving_default']

    # Create a Keras model from the concrete function
    keras_model = tf.keras.models.load_model(tf_model_path)

    # Save as Keras HDF5
    keras_path = os.path.join(output_dir, 'model.h5')
    keras_model.save(keras_path)
    print(f"✅ Keras model saved to: {keras_path}")

    return keras_path

if __name__ == "__main__":
    onnx_path = "../mobilenet_v1_ecg_model.onnx"
    output_dir = "tfjs_model"

    try:
        create_tfjs_model_from_onnx(onnx_path, output_dir)
        print("🎉 Conversion completed successfully!")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
