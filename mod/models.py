#!/usr/bin/env python3
"""
Complete model architectures for ECG classification
Includes MobileNet, CNN-LSTM, ResNet1D, Transformer, and Simple models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleCNN(nn.Module):
    """Simple CNN model for ECG classification - Optimized Fallback"""
    
    def __init__(self, input_size=1000, num_classes=5, **kwargs):
        super(SimpleCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Accept width_multiplier for compatibility (ignored for this simple model)
        width_multiplier = kwargs.get('width_multiplier', 1.0)
        
        # CNN layers - optimized for 1000 input size
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size)
            dummy_output = self._forward_features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"SimpleCNN: {total_params:,} parameters (~{total_params*4/1024/1024:.2f}MB)")
    
    def _forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool1d(x, 1)
        return x
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_size)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNLSTMModel(nn.Module):
    """CNN-LSTM model for ECG classification"""
    
    def __init__(self, input_size=1000, num_classes=5, **kwargs):
        super(CNNLSTMModel, self).__init__()
        
        # CNN feature extractor
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        
        # LSTM
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        
        # Classifier
        self.fc = nn.Linear(256, num_classes)  # 256 = 128*2 (bidirectional)
        self.dropout = nn.Dropout(0.5)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"CNN-LSTM: {total_params:,} parameters (~{total_params*4/1024/1024:.2f}MB)")
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        # CNN features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 3, stride=2, padding=1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 3, stride=2, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 3, stride=2, padding=1)
        
        # Prepare for LSTM
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last time step
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNet1DBlock(nn.Module):
    """1D ResNet block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super(ResNet1DBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ResNet1DClassifier(nn.Module):
    """ResNet-based ECG classifier"""
    
    def __init__(self, input_size=1000, num_classes=5, **kwargs):
        super(ResNet1DClassifier, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ResNet1D: {total_params:,} parameters (~{total_params*4/1024/1024:.2f}MB)")
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResNet1DBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResNet1DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TransformerECGClassifier(nn.Module):
    """Transformer-based ECG classifier"""
    
    def __init__(self, input_size=1000, num_classes=5, d_model=128, nhead=8, num_layers=4, **kwargs):
        super(TransformerECGClassifier, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=input_size)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Transformer: {total_params:,} parameters (~{total_params*4/1024/1024:.2f}MB)")
    
    def forward(self, x):
        # x: (batch_size, input_size)
        x = x.unsqueeze(-1)  # (batch_size, input_size, 1)
        x = self.input_projection(x)  # (batch_size, input_size, d_model)
        x = self.pos_encoding(x)
        
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

# MobileNet classes for real-time ECG classification

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for 1D signals"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                  stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNetV1_1D(nn.Module):
    """MobileNetV1 adapted for 1D ECG signals"""
    
    def __init__(self, input_size=1000, num_classes=5, width_multiplier=1.0, **kwargs):
        super().__init__()
        
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        base_channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        channels = [make_divisible(c * width_multiplier) for c in base_channels]
        
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv1d(1, channels[0], 3, stride=2, padding=1, bias=False),
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
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(channels[13], num_classes)
        )
        
        self._initialize_weights()
        
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
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x

class MobileNetLite(nn.Module):
    """Ultra-lightweight MobileNet for maximum speed"""
    
    def __init__(self, input_size=1000, num_classes=5, **kwargs):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv1d(16, 32, 3, stride=1, padding=1),
            DepthwiseSeparableConv1d(32, 64, 3, stride=2, padding=1),
            DepthwiseSeparableConv1d(64, 128, 3, stride=2, padding=1),
            DepthwiseSeparableConv1d(128, 256, 3, stride=2, padding=1),
            DepthwiseSeparableConv1d(256, 512, 3, stride=2, padding=1),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MobileNetLite: {total_params:,} parameters (~{total_params*4/1024:.1f}KB)")
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x

class ECGAttention(nn.Module):
    """Attention-based model for ECG classification"""
    
    def __init__(self, input_size=1000, num_classes=5, hidden_size=128, num_heads=4, **kwargs):
        super(ECGAttention, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Feature extraction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, batch_first=True)
        
        # Classifier
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ECGAttention: {total_params:,} parameters (~{total_params*4/1024/1024:.2f}MB)")
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, 64)
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.mean(dim=1)  # Global average pooling
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ECGHybrid(nn.Module):
    """Hybrid CNN-LSTM model for ECG classification"""
    
    def __init__(self, input_size=1000, num_classes=5, lstm_units=128, **kwargs):
        super(ECGHybrid, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        
        # CNN feature extractor
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5)
        self.bn2 = nn.BatchNorm1d(64)
        
        # LSTM
        self.lstm = nn.LSTM(64, lstm_units, batch_first=True, bidirectional=True)
        
        # Classifier
        self.fc = nn.Linear(lstm_units * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.4)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ECGHybrid: {total_params:,} parameters (~{total_params*4/1024/1024:.2f}MB)")
    
    def forward(self, x):
        # x shape: (batch_size, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # CNN features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        
        # Prepare for LSTM
        x = x.transpose(1, 2)  # (batch_size, seq_len, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last time step
        x = lstm_out[:, -1, :]
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Model registry - THIS IS THE KEY PART THAT WAS MISSING
models = {
    'simple': SimpleCNN,
    'cnn_lstm': CNNLSTMModel,
    'resnet1d': ResNet1DClassifier,
    'transformer': TransformerECGClassifier,
    'attention': ECGAttention,
    'hybrid': ECGHybrid,
    'mobilenet': MobileNetV1_1D,
    'mobilenet_lite': MobileNetLite,
    'mobilenet_0.75': lambda **kwargs: MobileNetV1_1D(width_multiplier=0.75, **kwargs),
    'mobilenet_0.5': lambda **kwargs: MobileNetV1_1D(width_multiplier=0.5, **kwargs),
}

def create_model(model_name, **kwargs):
    """Create model by name with parameters"""
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    
    model_class = models[model_name]
    return model_class(**kwargs)

def list_models():
    """List all available models"""
    return list(models.keys())

def get_model_info():
    """Get information about all models"""
    info = {}
    for name, model_class in models.items():
        if callable(model_class) and hasattr(model_class, '__name__'):
            info[name] = {
                'class': model_class.__name__,
                'description': model_class.__doc__ or "No description"
            }
        else:
            info[name] = {
                'class': 'Lambda function',
                'description': 'Variant of another model'
            }
    return info

# Test function
if __name__ == "__main__":
    print("Testing all ECG models...")
    
    input_size = 1000
    num_classes = 5
    batch_size = 2
    
    for model_name in models.keys():
        try:
            print(f"\nTesting {model_name}...")
            model = create_model(model_name, input_size=input_size, num_classes=num_classes)
            dummy_input = torch.randn(batch_size, input_size)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"✅ {model_name}: Input {dummy_input.shape} → Output {output.shape}")
            
        except Exception as e:
            print(f"❌ {model_name}: {e}")
    
    print(f"\nAvailable models: {list_models()}")