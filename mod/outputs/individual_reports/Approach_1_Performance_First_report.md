# Approach_1_Performance_First - Detailed Analysis Report

**Generated**: 2025-08-31 17:49:24
**Model Path**: outputs/models/approach1_performance_first/mobilenet_best.pth

## Model Overview

### Basic Information
- **Model Name**: Approach_1_Performance_First
- **Training Completed**: Epoch 42
- **Model Size**: 4.42 MB
- **Total Parameters**: 1,158,696
- **Timestamp**: 2025-08-30T14:57:23.498176

### Architecture Configuration
- **Width Multiplier**: 0.6
- **Dropout Rate**: 0.45
- **Batch Size**: 32
- **Learning Rate**: 0.0008
- **Weight Decay**: 0.0006

## Performance Metrics

### Validation Performance
- **Validation Accuracy**: 95.67%
- **Validation F1**: 95.70%
- **Balanced Accuracy**: 93.81%

### Test Set Performance
- **Test Accuracy**: 95.52%
- **Test F1 (Weighted)**: 95.56%
- **Test F1 (Macro)**: 92.01%
- **Test Precision**: 95.66%
- **Test Recall**: 95.52%
- **Balanced Accuracy**: 93.77%

## Generalization Analysis

### Gap Metrics
- **Generalization Gap**: 3.70%
- **Accuracy Gap**: 0.00%
- **F1 Gap**: 0.00%

### Gap Compliance Check
- **Generalization Gap < 5%**: PASS (3.70%)
- **Accuracy Gap 2-5%**: FAIL (0.00%)
- **F1 Gap 1-4%**: FAIL (0.00%)

## Performance Specifications

### Speed & Size
- **Average Inference Time**: 0.38 ms/sample
- **Model Size**: 4.42 MB
- **Parameter Count**: 1,158,696

### Deployment Readiness
- **Real-time Ready (<50ms)**: YES (0.38ms)
- **Mobile Ready (<5MB)**: YES (4.42MB)
- **Clinical Grade (>90% acc)**: YES (95.67%)

## Per-Class Performance Analysis

| Class | Precision (%) | Recall (%) | F1-Score (%) | Support |
|-------|---------------|------------|--------------|---------|
| F (Fusion) | 88.2 | 98.1 | 92.9 | 2250 |
| N (Normal) | 98.3 | 96.6 | 97.4 | 25094 |
| Q (Unknown) | 89.2 | 89.0 | 89.1 | 3279 |
| S (Supraventricular) | 84.1 | 91.7 | 87.8 | 1800 |
| V (Ventricular) | 92.4 | 93.4 | 92.9 | 2250 |


## Training Configuration Details

### Core Training Parameters
- **Epochs Trained**: 42
- **Batch Size**: 32
- **Learning Rate**: 0.0008
- **Weight Decay**: 0.0006

### Regularization Settings
- **Dropout Rate**: 0.45
- **Label Smoothing**: 0.12
- **Gradient Clip Norm**: 0.8

### Architecture Settings
- **Width Multiplier**: 0.6
- **Use Mixed Precision**: True

### Training Strategy
- **Early Stopping Metric**: val_f1
- **Early Stopping Patience**: 20
- **Scheduler Factor**: 0.5

## Deployment Assessment

### Mobile Deployment Readiness
READY FOR MOBILE

- Size Requirement: Met (4.42MB < 5MB)
- Speed Requirement: Met (0.38ms < 50ms)
- Accuracy Requirement: Met (95.67% > 90%)

### Clinical Deployment Readiness
READY FOR CLINICAL USE

- Clinical Accuracy: Excellent (95.67%)
- Generalization: Excellent (3.70% gap)
- Class Balance: Good (93.8% bal acc)
---
**Report Generated**: 2025-08-31 17:49:24
