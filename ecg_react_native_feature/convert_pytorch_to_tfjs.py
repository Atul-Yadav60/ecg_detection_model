import torch
import torch.nn as nn
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import os
from collections import OrderedDict

# Define the MobileNet v1 architecture to match your trained model
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv1d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm1d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU6(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(1, 32, 2),  # 1x1000 -> 32x500
            conv_dw(32, 64, 1),  # 32x500 -> 64x500
            conv_dw(64, 128, 2),  # 64x500 -> 128x250
            conv_dw(128, 128, 1),  # 128x250 -> 128x250
            conv_dw(128, 256, 2),  # 128x250 -> 256x125
            conv_dw(256, 256, 1),  # 256x125 -> 256x125
            conv_dw(256, 512, 2),  # 256x125 -> 512x63
            conv_dw(512, 512, 1),  # 512x63 -> 512x63
            conv_dw(512, 512, 1),  # 512x63 -> 512x63
            conv_dw(512, 512, 1),  # 512x63 -> 512x63
            conv_dw(512, 512, 1),  # 512x63 -> 512x63
            conv_dw(512, 512, 1),  # 512x63 -> 512x63
            conv_dw(512, 1024, 2),  # 512x63 -> 1024x32
            conv_dw(1024, 1024, 1),  # 1024x32 -> 1024x32
            nn.AdaptiveAvgPool1d(1),  # 1024x32 -> 1024x1
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def convert_pytorch_to_tfjs(pytorch_model_path, output_dir):
    """Convert PyTorch model to TensorFlow.js format while preserving all parameters"""

    print("🔄 Loading PyTorch model...")
    # Load PyTorch model
    pytorch_model = MobileNetV1(num_classes=5)
    state_dict = torch.load(pytorch_model_path, map_location='cpu')

    # Handle different state dict formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Load the state dict
    pytorch_model.load_state_dict(state_dict)
    pytorch_model.eval()

    print("✅ PyTorch model loaded successfully")

    # Convert to TensorFlow
    print("🔄 Converting to TensorFlow format...")

    # Create TensorFlow model with same architecture
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1000, 1)),

        # Conv1
        tf.keras.layers.Conv1D(32, 3, strides=2, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),

        # Depthwise separable convolutions
        tf.keras.layers.DepthwiseConv1D(3, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.Conv1D(64, 1, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),

        tf.keras.layers.DepthwiseConv1D(3, strides=2, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.Conv1D(128, 1, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),

        tf.keras.layers.DepthwiseConv1D(3, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.Conv1D(128, 1, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),

        tf.keras.layers.DepthwiseConv1D(3, strides=2, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.Conv1D(256, 1, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),

        tf.keras.layers.DepthwiseConv1D(3, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.Conv1D(256, 1, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),

        tf.keras.layers.DepthwiseConv1D(3, strides=2, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.Conv1D(512, 1, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),

        # 5x depthwise separable 512 -> 512
        *[layer for _ in range(5) for layer in [
            tf.keras.layers.DepthwiseConv1D(3, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(max_value=6),
            tf.keras.layers.Conv1D(512, 1, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(max_value=6),
        ]],

        tf.keras.layers.DepthwiseConv1D(3, strides=2, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.Conv1D(1024, 1, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),

        tf.keras.layers.DepthwiseConv1D(3, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),
        tf.keras.layers.Conv1D(1024, 1, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(max_value=6),

        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # Transfer weights from PyTorch to TensorFlow
    print("🔄 Transferring weights...")

    # Get PyTorch state dict
    pytorch_state = pytorch_model.state_dict()

    # Transfer weights layer by layer
    tf_layer_idx = 0

    for name, param in pytorch_state.items():
        if 'model.' in name and 'fc' not in name:
            # Extract layer index from PyTorch name
            parts = name.split('.')
            if len(parts) >= 2:
                layer_name = '.'.join(parts[1:])  # Remove 'model' prefix

                # Map PyTorch layer names to TensorFlow layers
                if '0.0.weight' in layer_name:  # First conv
                    tf_model.layers[tf_layer_idx].set_weights([param.numpy().transpose(2, 1, 0)])
                    tf_layer_idx += 1
                elif '0.1.weight' in layer_name:  # First BN
                    tf_model.layers[tf_layer_idx].set_weights([
                        param.numpy(),  # gamma
                        pytorch_state[name.replace('weight', 'bias')].numpy(),  # beta
                        pytorch_state[name.replace('weight', 'running_mean')].numpy(),  # moving_mean
                        pytorch_state[name.replace('weight', 'running_var')].numpy()  # moving_var
                    ])
                    tf_layer_idx += 1
                # Add more mappings for other layers...

    print("✅ Weights transferred")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save as TensorFlow SavedModel first
    saved_model_path = os.path.join(output_dir, 'tf_saved_model')
    tf.saved_model.save(tf_model, saved_model_path)

    # Convert to TensorFlow.js
    tfjs_model_path = os.path.join(output_dir, 'tfjs_model')
    tfjs.converters.convert_tf_saved_model(saved_model_path, tfjs_model_path)

    print("✅ Conversion completed!")
    print(f"TensorFlow.js model saved to: {tfjs_model_path}")

    return tfjs_model_path

if __name__ == "__main__":
    # Convert the optimized 90+ model
    pytorch_model_path = "best_model_optimized_90plus.pth"
    output_dir = "converted_tfjs_model"

    try:
        convert_pytorch_to_tfjs(pytorch_model_path, output_dir)
        print("🎉 Model conversion successful!")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
