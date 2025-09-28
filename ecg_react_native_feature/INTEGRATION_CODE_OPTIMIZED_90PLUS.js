// React Native Integration Code for Optimized 90+ Model
// Using ONNX.js for maximum performance and compatibility

import { InferenceSession } from 'onnxjs-react-native';

class ECGMLAnalyzer {
  session = null;
  isModelLoaded = false;
  modelPath = 'assets/ml_models/mobilenet_v1_ecg_optimized_90plus.onnx';
  
  // Model configuration for optimized 90+ model
  MODEL_CONFIG = {
    inputShape: [1000, 1],
    outputClasses: 5,
    segmentLength: 1000,
    accuracy: '90%+',
    framework: 'ONNX.js'
  };
  
  async loadModel() {
    try {
      console.log('Loading Optimized 90+ ECG model...');
      
      this.session = new InferenceSession();
      await this.session.loadModel(this.modelPath);
      
      this.isModelLoaded = true;
      console.log('Optimized 90+ model loaded successfully');
      console.log('Model parameters: 1,145,597');
      console.log('Target accuracy: 90%+');
      
      return true;
    } catch (error) {
      console.error('Model loading failed:', error);
      this.isModelLoaded = false;
      return false;
    }
  }
  
  async analyzeECG(ecgData) {
    if (!this.session || !this.isModelLoaded) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }
    
    const startTime = performance.now();
    
    try {
      // Preprocess ECG data to 1000 samples
      const processedData = this.preprocessECG(ecgData);
      
      // Create input tensor (1, 1, 1000)
      const inputTensor = new Float32Array(processedData);
      
      // Run inference
      const outputMap = await this.session.run({'ecg_input': inputTensor});
      const probabilities = outputMap.get('classification_output');
      
      const processingTime = performance.now() - startTime;
      
      // Interpret results
      const result = this.interpretResults(probabilities);
      result.processingTime = processingTime;
      
      console.log(`ECG Analysis: ${result.condition} (${(result.confidence * 100).toFixed(1)}%) in ${processingTime.toFixed(1)}ms`);
      
      return result;
      
    } catch (error) {
      console.error('ECG analysis failed:', error);
      throw error;
    }
  }
  
  preprocessECG(rawData) {
    // Ensure exactly 1000 samples
    let processed = [...rawData];
    
    if (processed.length !== 1000) {
      processed = this.resampleTo1000(processed);
    }
    
    // Apply z-score normalization
    processed = this.normalizeZScore(processed);
    
    return processed;
  }
  
  resampleTo1000(data) {
    if (data.length === 1000) return data;
    
    const resampled = [];
    const step = (data.length - 1) / 999;
    
    for (let i = 0; i < 1000; i++) {
      const index = i * step;
      const lower = Math.floor(index);
      const upper = Math.min(Math.ceil(index), data.length - 1);
      const fraction = index - lower;
      
      if (lower === upper) {
        resampled.push(data[lower]);
      } else {
        const interpolated = data[lower] * (1 - fraction) + data[upper] * fraction;
        resampled.push(interpolated);
      }
    }
    
    return resampled;
  }
  
  normalizeZScore(data) {
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
    const std = Math.sqrt(variance);
    
    if (std === 0) return data.map(() => 0);
    
    return data.map(val => (val - mean) / std);
  }
  
  interpretResults(probabilities) {
    const probs = Array.from(probabilities);
    const maxIndex = probs.indexOf(Math.max(...probs));
    const confidence = probs[maxIndex];
    
    const classLabels = ['Normal', 'Ventricular', 'Supraventricular', 'Fusion', 'Unknown'];
    const condition = classLabels[maxIndex];
    
    return {
      condition,
      confidence,
      probabilities: {
        Normal: probs[0],
        Ventricular: probs[1],
        Supraventricular: probs[2],
        Fusion: probs[3],
        Unknown: probs[4]
      },
      isReliable: confidence >= 0.8,
      clinicalGrade: confidence >= 0.9
    };
  }
  
  getModelInfo() {
    return {
      isLoaded: this.isModelLoaded,
      modelName: 'MobileNet v1 ECG Optimized 90+',
      version: '2.1.0',
      accuracy: '90%+',
      parameters: '1,145,597',
      framework: 'ONNX.js',
      optimized: true,
      deploymentReady: true
    };
  }
  
  dispose() {
    if (this.session) {
      this.session = null;
      this.isModelLoaded = false;
      console.log('ECG model disposed');
    }
  }
}

export default ECGMLAnalyzer;
