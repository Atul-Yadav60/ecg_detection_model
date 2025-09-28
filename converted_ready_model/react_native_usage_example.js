// React Native Usage Example
// Install: npm install @tensorflow/tfjs-react-native @tensorflow/tfjs-platform-react-native

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import '@tensorflow/tfjs-platform-react-native';

class ECGClassifierRN {
    constructor() {
        this.model = null;
    }
    
    async loadModel() {
        // Load the converted model
        this.model = await tf.loadLayersModel('file:///path/to/converted_ready_model/model.json');
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

// Usage in React Native component
import React, { useEffect, useState } from 'react';
import { View, Text, Button } from 'react-native';

const ECGApp = () => {
    const [classifier, setClassifier] = useState(null);
    const [result, setResult] = useState(null);
    
    useEffect(() => {
        const initClassifier = async () => {
            const cls = new ECGClassifierRN();
            await cls.loadModel();
            setClassifier(cls);
        };
        initClassifier();
    }, []);
    
    const handlePredict = async () => {
        if (classifier) {
            // Example ECG signal (1000 samples)
            const ecgSignal = new Array(1000).fill(0).map(() => Math.random() * 2 - 1);
            const prediction = await classifier.predict(ecgSignal);
            setResult(prediction);
        }
    };
    
    return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <Button title="Predict ECG" onPress={handlePredict} />
            {result && (
                <View>
                    <Text>Predicted: {result.predictedClass}</Text>
                    <Text>Confidence: {(result.confidence * 100).toFixed(2)}%</Text>
                </View>
            )}
        </View>
    );
};

export default ECGApp;
