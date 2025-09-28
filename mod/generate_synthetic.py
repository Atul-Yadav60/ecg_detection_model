#!/usr/bin/env python3
"""
Generate synthetic ECG data for minority classes
"""

import numpy as np
import scipy.signal as signal
from scipy import interpolate

def generate_synthetic_ecg(ecg_type='normal', length=5000):
    """Generate synthetic ECG signals"""
    
    if ecg_type == 'normal':
        # Generate normal ECG pattern
        t = np.linspace(0, length/500, length)
        
        # Create P wave
        p_wave = 0.1 * np.exp(-((t - 0.2) ** 2) / 0.001)
        
        # Create QRS complex
        qrs = 1.0 * np.exp(-((t - 0.4) ** 2) / 0.0001)
        
        # Create T wave
        t_wave = 0.3 * np.exp(-((t - 0.6) ** 2) / 0.002)
        
        # Combine waves
        synthetic_ecg = p_wave + qrs + t_wave
        
        # Add baseline wander
        baseline = 0.05 * np.sin(2 * np.pi * 0.1 * t)
        synthetic_ecg += baseline
        
        return synthetic_ecg
    
    elif ecg_type == 'ventricular':
        # Generate ventricular arrhythmia pattern
        t = np.linspace(0, length/500, length)
        
        # Irregular QRS complexes
        qrs_times = np.array([0.4, 0.8, 1.3, 1.9, 2.6, 3.4, 4.3, 5.3])
        synthetic_ecg = np.zeros(length)
        
        for qrs_time in qrs_times:
            if qrs_time < length/500:
                idx = int(qrs_time * 500)
                if idx < length:
                    synthetic_ecg[idx] = 1.5
        
        # Smooth the signal
        synthetic_ecg = signal.savgol_filter(synthetic_ecg, 51, 3)
        
        return synthetic_ecg
    
    elif ecg_type == 'supraventricular':
        # Generate supraventricular arrhythmia
        t = np.linspace(0, length/500, length)
        
        # Rapid P waves
        p_wave_times = np.arange(0.2, length/500, 0.3)
        synthetic_ecg = np.zeros(length)
        
        for p_time in p_wave_times:
            if p_time < length/500:
                idx = int(p_time * 500)
                if idx < length:
                    synthetic_ecg[idx] = 0.2
        
        # Add QRS complexes
        qrs_times = np.arange(0.4, length/500, 0.6)
        for qrs_time in qrs_times:
            if qrs_time < length/500:
                idx = int(qrs_time * 500)
                if idx < length:
                    synthetic_ecg[idx] = 1.0
        
        # Smooth the signal
        synthetic_ecg = signal.savgol_filter(synthetic_ecg, 51, 3)
        
        return synthetic_ecg
    
    return np.zeros(length)

def generate_synthetic_dataset():
    """Generate synthetic dataset for minority classes"""
    
    print("🧬 Generating synthetic ECG data...")
    
    synthetic_segments = []
    synthetic_labels = []
    
    # Generate synthetic data for each minority class
    class_configs = {
        'F': {'type': 'fusion', 'count': 4000},
        'S': {'type': 'supraventricular', 'count': 3000},
        'V': {'type': 'ventricular', 'count': 2000}
    }
    
    for label, config in class_configs.items():
        print(f"Generating {config['count']} synthetic samples for class {label}...")
        
        for i in range(config['count']):
            # Add some randomness
            noise_level = np.random.uniform(0.01, 0.05)
            length_variation = np.random.uniform(0.9, 1.1)
            
            # Generate synthetic signal
            synthetic_signal = generate_synthetic_ecg(config['type'], 5000)
            
            # Add noise
            noise = np.random.normal(0, noise_level, len(synthetic_signal))
            synthetic_signal += noise
            
            # Vary length slightly
            if length_variation != 1.0:
                new_length = int(len(synthetic_signal) * length_variation)
                synthetic_signal = signal.resample(synthetic_signal, new_length)
                # Pad or truncate to original length
                if len(synthetic_signal) > 5000:
                    synthetic_signal = synthetic_signal[:5000]
                else:
                    synthetic_signal = np.pad(synthetic_signal, (0, 5000 - len(synthetic_signal)))
            
            synthetic_segments.append(synthetic_signal)
            synthetic_labels.append(label)
    
    synthetic_segments = np.array(synthetic_segments)
    synthetic_labels = np.array(synthetic_labels)
    
    # Save synthetic data
    np.save('data/processed/train/segments_synthetic.npy', synthetic_segments)
    np.save('data/processed/train/labels_synthetic.npy', synthetic_labels)
    
    print(f"✅ Generated {len(synthetic_segments)} synthetic samples")
    
    return synthetic_segments, synthetic_labels

if __name__ == "__main__":
    generate_synthetic_dataset()
