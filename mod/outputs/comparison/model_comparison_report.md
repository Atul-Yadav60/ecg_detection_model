# 3-Way MobileNetV1 1D ECG Model Comparison

**Generated**: 2025-08-31 17:41:20

## Rankings

1st **Approach 2 Backup** (Score: 82/100)
- Accuracy: 94.23%, Gap: 2.27%, Size: 3.7MB

2nd **Approach 1** (Score: 75/100)
- Accuracy: 95.67%, Gap: 3.70%, Size: 4.4MB

3rd **Approach 2 Current** (Score: 74/100)
- Accuracy: 89.71%, Gap: 3.51%, Size: 3.7MB

## Complete Comparison

| Metric | Approach 1 | Approach 2 Current | Approach 2 Backup |
|--------|------------|-------------------|-------------------|
| **Validation Accuracy (%)** | 95.67 | 89.71 | 94.23 |
| **Validation F1 (%)** | 95.70 | 90.29 | 94.37 |
| **Generalization Gap (%)** | 3.70 | 3.51 | 2.27 |
| **Accuracy Gap (%)** | 0.00 | 3.51 | 2.27 |
| **F1 Gap (%)** | 0.00 | 2.96 | 2.12 |
| **Model Size (MB)** | 4.42 | 3.73 | 3.73 |
| **Inference Time (ms)** | 0.38 | 0.85 | 0.41 |
| **Training Epoch** | 42.00 | 42.00 | 38.00 |
| **Parameters** | 1,158,696 | 976,864 | 976,864 |


## Gap Requirements Analysis

**Your Requirements**: Gen<5%, Acc=2-5%, F1=1-4%

### Approach 1:
- Gen Gap: 3.70% PASS
- Acc Gap: 0.00% FAIL
- F1 Gap: 0.00% FAIL

### Approach 2 Current:
- Gen Gap: 3.51% PASS
- Acc Gap: 3.51% PASS
- F1 Gap: 2.96% PASS

### Approach 2 Backup:
- Gen Gap: 2.27% PASS
- Acc Gap: 2.27% PASS
- F1 Gap: 2.12% PASS

## Deployment Recommendation

**Recommended Model**: Approach 2 Backup

**Reasons**:
- Highest overall score: 82/100
- Accuracy: 94.23%
- Generalization Gap: 2.27%
- Mobile Ready
- Real-time Ready

