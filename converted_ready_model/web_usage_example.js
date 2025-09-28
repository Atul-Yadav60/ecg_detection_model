// Web Usage Example (JavaScript/TypeScript)
// Install: npm install @tensorflow/tfjs

import * as tf from '@tensorflow/tfjs';

class ECGClassifier {
    constructor() {
        this.model = null;
    }
    
    async loadModel() {
        // Load the converted model
        this.model = await tf.loadLayersModel('/path/to/converted_ready_model/model.json');
        console.log('Model loaded successfully');
    }
    
    preprocessECG(ecgSignal) {
        // Ensure 1000 samples
        if (ecgSignal.length !== 1000) {
            // Resample to 1000 samples
            const resampled = [];
            for (let i = 0; i < 1000; i++) {
                const index = (i / 999) * (ecgSignal.length - 1);
                const lower = Math.floor(index);
                const upper = Math.ceil(index);
                const weight = index - lower;
                resampled[i] = ecgSignal[lower] * (1 - weight) + ecgSignal[upper] * weight;
            }
            ecgSignal = resampled;
        }
        
        // Z-score normalization
        const mean = ecgSignal.reduce((a, b) => a + b, 0) / ecgSignal.length;
        const variance = ecgSignal.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / ecgSignal.length;
        const std = Math.sqrt(variance);
        
        if (std === 0) {
            return new Array(1000).fill(0);
        }
        
        return ecgSignal.map(x => (x - mean) / std);
    }
    
    async predict(ecgSignal) {
        if (!this.model) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }
        
        // Preprocess
        const processed = this.preprocessECG(ecgSignal);
        
        // Convert to tensor
        const input = tf.tensor3d([processed], [1, 1000, 1]);
        
        // Predict
        const prediction = this.model.predict(input);
        const result = await prediction.data();
        
        // Clean up
        input.dispose();
        prediction.dispose();
        
        // Return class probabilities
        const classes = ['Normal', 'AFib', 'Other', 'Noise', 'Unclassified'];
        const probabilities = Array.from(result);
        const maxIndex = probabilities.indexOf(Math.max(...probabilities));
        
        return {
            predictedClass: classes[maxIndex],
            confidence: probabilities[maxIndex],
            allProbabilities: probabilities.map((prob, index) => ({
                class: classes[index],
                probability: prob
            }))
        };
    }
}

// Usage
const classifier = new ECGClassifier();
await classifier.loadModel();

// Example ECG signal (1000 samples)
const ecgSignal = new Array(1000).fill(0).map(() => Math.random() * 2 - 1);
const result = await classifier.predict(ecgSignal);
console.log('Prediction:', result);
