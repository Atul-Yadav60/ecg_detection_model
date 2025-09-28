#!/usr/bin/env python3
"""
Convert TensorFlow SavedModel to TensorFlow.js format
Alternative approach using Python API directly
"""

import os
import sys
import tensorflow as tf
from tensorflowjs.converters import tf_saved_model_conversion_v2

def convert_savedmodel_to_tfjs():
    """Convert TensorFlow SavedModel to TensorFlow.js format"""
    
    print("🔄 Converting TensorFlow SavedModel to TensorFlow.js...")
    
    # Paths
    saved_model_dir = "tf_saved_model"
    tfjs_output_dir = "tfjs_model"
    
    # Check if SavedModel exists
    if not os.path.exists(saved_model_dir):
        print(f"❌ SavedModel directory not found: {saved_model_dir}")
        return False
    
    # Create output directory
    os.makedirs(tfjs_output_dir, exist_ok=True)
    
    try:
        # Convert using TensorFlow.js converter
        print(f"📂 Input: {saved_model_dir}")
        print(f"📂 Output: {tfjs_output_dir}")
        
        # Use the converter function directly
        tf_saved_model_conversion_v2.convert_tf_saved_model(
            saved_model_dir,
            tfjs_output_dir,
            input_format='tf_saved_model',
            output_format='tfjs_graph_model',
            quantization_dtype=None,
            skip_op_check=False,
            strip_debug_ops=True,
            weight_shard_size_bytes=4194304
        )
        
        print("✅ Conversion completed successfully!")
        
        # List generated files
        if os.path.exists(tfjs_output_dir):
            files = os.listdir(tfjs_output_dir)
            print(f"📁 Generated files in {tfjs_output_dir}:")
            for file in files:
                file_path = os.path.join(tfjs_output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    print(f"   - {file} ({size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def verify_tfjs_model():
    """Verify the converted TensorFlow.js model"""
    
    tfjs_dir = "tfjs_model"
    
    if not os.path.exists(tfjs_dir):
        print("❌ TensorFlow.js model directory not found")
        return False
    
    # Check for required files
    required_files = ["model.json"]
    files = os.listdir(tfjs_dir)
    
    print(f"📁 Files in {tfjs_dir}:")
    for file in files:
        print(f"   - {file}")
    
    if "model.json" in files:
        print("✅ TensorFlow.js model.json found")
        return True
    else:
        print("❌ model.json not found")
        return False

if __name__ == "__main__":
    print("🎯 Converting TensorFlow SavedModel to TensorFlow.js")
    print("=" * 50)
    
    # Convert the model
    success = convert_savedmodel_to_tfjs()
    
    if success:
        print("\n🔍 Verifying conversion...")
        verify_success = verify_tfjs_model()
        
        if verify_success:
            print("\n🎉 CONVERSION COMPLETE!")
            print("✅ TensorFlow.js model ready for React Native")
            print("\n📱 Next steps:")
            print("1. Copy tfjs_model/ folder to your React Native assets")
            print("2. Update ECGMLAnalyzer.ts to use TensorFlow.js model")
            print("3. Test model loading in React Native app")
        else:
            print("\n⚠️ Conversion completed but verification failed")
    else:
        print("\n❌ Conversion failed")
        print("💡 Try alternative conversion methods")
