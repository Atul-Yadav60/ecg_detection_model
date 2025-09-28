import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for 1D signals"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                  stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNetV1_1D(nn.Module):
    """
    MobileNetV1 adapted for 1D ECG signals
    Optimized for real-time smartphone inference
    ~350,000 parameters (~1.4MB) - Perfect balance!
    """
    
    def __init__(self, input_size=1000, num_classes=5, width_multiplier=1.0):
        super().__init__()
        
        # Calculate channel sizes with width multiplier
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        base_channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        channels = [make_divisible(c * width_multiplier) for c in base_channels]
        
        # Feature extraction
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv1d(1, channels[0], 3, stride=2, padding=1),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            
            # Depthwise separable blocks
            DepthwiseSeparableConv1d(channels[0], channels[1], 3, stride=1, padding=1),
            DepthwiseSeparableConv1d(channels[1], channels[2], 3, stride=2, padding=1),
            DepthwiseSeparableConv1d(channels[2], channels[3], 3, stride=1, padding=1),
            DepthwiseSeparableConv1d(channels[3], channels[4], 3, stride=2, padding=1),
            DepthwiseSeparableConv1d(channels[4], channels[5], 3, stride=1, padding=1),
            DepthwiseSeparableConv1d(channels[5], channels[6], 3, stride=2, padding=1),
            DepthwiseSeparableConv1d(channels[6], channels[7], 3, stride=1, padding=1),
            DepthwiseSeparableConv1d(channels[7], channels[8], 3, stride=1, padding=1),
            DepthwiseSeparableConv1d(channels[8], channels[9], 3, stride=1, padding=1),
            DepthwiseSeparableConv1d(channels[9], channels[10], 3, stride=1, padding=1),
            DepthwiseSeparableConv1d(channels[10], channels[11], 3, stride=1, padding=1),
            DepthwiseSeparableConv1d(channels[11], channels[12], 3, stride=2, padding=1),
            DepthwiseSeparableConv1d(channels[12], channels[13], 3, stride=1, padding=1),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(channels[13], num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MobileNetV1-1D: {total_params:,} parameters (~{total_params*4/1024/1024:.2f}MB)")
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x shape: (batch_size, 1000)
        x = x.unsqueeze(1)  # Add channel dim: (batch_size, 1, 1000)
        x = self.features(x)
        x = x.squeeze(-1)   # Remove last dim: (batch_size, 1024)
        x = self.classifier(x)
        return x

class MobileNetV1_1D_Lite(nn.Module):
    """Lightweight version for faster inference"""
    
    def __init__(self, input_size=1000, num_classes=5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv1d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            # Depthwise separable blocks
            DepthwiseSeparableConv1d(16, 32, 3, stride=1, padding=1),
            DepthwiseSeparableConv1d(32, 64, 3, stride=2, padding=1),
            DepthwiseSeparableConv1d(64, 128, 3, stride=2, padding=1),
            DepthwiseSeparableConv1d(128, 256, 3, stride=2, padding=1),
            DepthwiseSeparableConv1d(256, 512, 3, stride=2, padding=1),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MobileNetV1-1D Lite: {total_params:,} parameters (~{total_params*4/1024:.1f}KB)")
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x

# Test the model
if __name__ == "__main__":
    # Test full version
    model = MobileNetV1_1D(input_size=1000, num_classes=5)
    
    # Test lite version  
    model_lite = MobileNetV1_1D_Lite(input_size=1000, num_classes=5)
    
    # Test inference
    dummy_input = torch.randn(1, 1000)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("MobileNetV1-1D ready for training!")