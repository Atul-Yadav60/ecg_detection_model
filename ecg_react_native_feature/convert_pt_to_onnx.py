#!/usr/bin/env python3
"""
Convert mobilenet_v1_ecg_focused_90.pt to ONNX format
"""

import torch
import torch.nn as nn
import os

# Model architecture
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, 
                                 stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNetV1_ECG_Focused90(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNetV1_ECG_Focused90, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(16, 40, stride=1),
            DepthwiseSeparableConv(40, 80, stride=2),
            DepthwiseSeparableConv(80, 80, stride=1),
            DepthwiseSeparableConv(80, 152, stride=2),
            DepthwiseSeparableConv(152, 152, stride=1),
            DepthwiseSeparableConv(152, 304, stride=2),
            DepthwiseSeparableConv(304, 304, stride=1),
            DepthwiseSeparableConv(304, 304, stride=1),
            DepthwiseSeparableConv(304, 304, stride=1),
            DepthwiseSeparableConv(304, 304, stride=1),
            DepthwiseSeparableConv(304, 304, stride=1),
            DepthwiseSeparableConv(304, 616, stride=2),
            DepthwiseSeparableConv(616, 616, stride=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(616, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def convert_pt_to_onnx():
    """Convert .pt file to ONNX format"""
    
    print('🔄 Converting mobilenet_v1_ecg_focused_90.pt to ONNX...')
    
    # Load the TorchScript model
    pt_path = 'assets/ml_models/mobilenet_v1_ecg_focused_90.pt'
    if not os.path.exists(pt_path):
        print(f'❌ TorchScript model not found: {pt_path}')
        return False
    
    try:
        # Load TorchScript model
        model = torch.jit.load(pt_path)
        model.eval()
        print('✅ TorchScript model loaded successfully')
        
        # Create dummy input
        dummy_input = torch.randn(1, 1, 1000)
        
        # Test model
        with torch.no_grad():
            test_output = model(dummy_input)
            print(f'✅ Model test successful - Output shape: {test_output.shape}')
        
        # Convert to ONNX
        onnx_path = 'assets/ml_models/mobilenet_v1_ecg_focused_90.onnx'
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['ecg_input'],
            output_names=['classification_output'],
            dynamic_axes={
                'ecg_input': {0: 'batch_size'},
                'classification_output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print(f'✅ ONNX conversion completed: {onnx_path}')
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print('✅ ONNX model verification passed')
        except Exception as e:
            print(f'⚠️ ONNX verification warning: {e}')
        
        return True
        
    except Exception as e:
        print(f'❌ Conversion failed: {e}')
        return False

if __name__ == "__main__":
    success = convert_pt_to_onnx()
    
    if success:
        print('\n🎉 TorchScript to ONNX conversion complete!')
    else:
        print('\n❌ Conversion failed')
