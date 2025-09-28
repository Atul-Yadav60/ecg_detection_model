#!/usr/bin/env python3
"""
Download additional ECG datasets to increase minority class samples
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import wfdb
import numpy as np
from tqdm import tqdm
import pandas as pd
from requests.adapters import HTTPAdapter, Retry
import argparse


def download_file(url, filename, chunk_size=8192, timeout=30, retries=5, verify=True):
    """Download file with retries, timeout, progress bar, and SSL fallback.

    - retries: total retry attempts on transient failures
    - timeout: per-request timeout in seconds
    - verify: SSL verification (set False as last-resort fallback)
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def _stream_download(_verify: bool):
        resp = session.get(url, stream=True, timeout=timeout, verify=_verify)
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))
        with open(filename, "wb") as f, tqdm(total=total_size or None, unit="B", unit_scale=True, desc=str(filename.name)) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                if total_size:
                    pbar.update(len(chunk))
        return True

    try:
        return _stream_download(verify)
    except requests.exceptions.SSLError as e:
        if verify:
            print(f"⚠️ SSL error for {url}. Retrying without certificate verification (last resort)...")
            try:
                return _stream_download(False)
            except Exception as e2:
                # Cleanup partial file
                if filename.exists() and filename.stat().st_size == 0:
                    try:
                        filename.unlink()
                    except Exception:
                        pass
                raise e2
        raise e
    except KeyboardInterrupt:
        print("\n⛔ Download interrupted by user.")
        # Cleanup partial file
        if filename.exists() and filename.stat().st_size == 0:
            try:
                filename.unlink()
            except Exception:
                pass
        raise


def download_mit_bih_additional(records=None, verify_ssl=True):
    """Download additional MIT-BIH records.

    records: optional list of record ids (e.g., ['200','201']). If None, uses default list.
    verify_ssl: whether to verify SSL certificates (True recommended).
    """
    print("📥 Downloading additional MIT-BIH records...")

    # Default records (complete MIT-BIH subset we target)
    default_records = [
        # Records with more ventricular beats
        '200', '201', '202', '203', '205', '207', '208', '209', '210',
        '212', '213', '214', '215', '217', '219', '220', '221', '222', '223',
        '228', '230', '231', '232', '233', '234',

        # Records with more supraventricular beats
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119',
        '121', '122', '123', '124'
    ]

    target_records = records or default_records

    base_url = "https://physionet.org/files/mitdb/1.0.0/"
    data_dir = Path("data/raw/additional_mit_bih")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Compute pending files to avoid re-downloading
    exts = [".dat", ".hea", ".atr"]
    pending = []
    for rec in target_records:
        for ext in exts:
            filepath = data_dir / f"{rec}{ext}"
            if not (filepath.exists() and filepath.stat().st_size > 0):
                pending.append((rec, ext))

    if not pending:
        print("✅ All requested records already present. Nothing to download.")
        return

    print(f"🧾 Pending files: {len(pending)} (records: {len(set(r for r,_ in pending))})")

    for record, ext in pending:
        url = f"{base_url}{record}{ext}"
        filename = data_dir / f"{record}{ext}"
        try:
            print(f"Downloading {record}{ext}...")
            download_file(url, filename, timeout=45, retries=5, verify=verify_ssl)
        except Exception as e:
            print(f"❌ HTTP download failed for {record}{ext}: {e}")
            # Fallback: try via wfdb (downloads by record base name without extension)
            try:
                print(f"↩️  Falling back to wfdb for record {record}...")
                wfdb.dl_database("mitdb", dl_dir=str(data_dir), records=[record])
            except Exception as e2:
                print(f"❌ wfdb fallback failed for {record}: {e2}")


def download_ptb_xl():
    """Download PTB-XL dataset (large ECG dataset)"""
    print("📥 Downloading PTB-XL dataset...")

    # PTB-XL is a large ECG dataset with good class distribution
    ptb_url = "https://physionet.org/files/ptb-xl/1.0.3/"

    # Note: PTB-XL requires registration and agreement
    print("⚠️  PTB-XL requires registration at: https://physionet.org/content/ptb-xl/1.0.3/")
    print("   Please download manually and place in data/raw/ptb_xl/")


def download_chapman():
    """Download Chapman dataset"""
    print("📥 Downloading Chapman ECG dataset...")

    # Chapman dataset has good arrhythmia distribution
    chapman_url = "https://physionet.org/files/chapman-shaoxing/1.0.0/"

    print("⚠️  Chapman dataset requires registration at: https://physionet.org/content/chapman-shaoxing/1.0.0/")
    print("   Please download manually and place in data/raw/chapman/")


def download_georgia():
    """Download Georgia dataset"""
    print("📥 Downloading Georgia ECG dataset...")

    # Georgia dataset has good class balance
    georgia_url = "https://physionet.org/files/georgia-12lead-ecg-challenge-database/1.0.0/"

    print("⚠️  Georgia dataset requires registration at: https://physionet.org/content/georgia-12lead-ecg-challenge-database/1.0.0/")
    print("   Please download manually and place in data/raw/georgia/")


def create_data_augmentation_script():
    """Create script for data augmentation"""

    script_content = '''#!/usr/bin/env python3
"""
ECG Data Augmentation for Minority Classes
"""

import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import random

def augment_ecg_signal(ecg_signal, augmentation_type='noise'):
    """Augment ECG signal using various techniques"""
    
    if augmentation_type == 'noise':
        # Add Gaussian noise
        noise_level = 0.01
        noise = np.random.normal(0, noise_level, ecg_signal.shape)
        return ecg_signal + noise
    
    elif augmentation_type == 'time_warp':
        # Time warping
        time_steps = len(ecg_signal)
        warp_factor = random.uniform(0.8, 1.2)
        new_time_steps = int(time_steps * warp_factor)
        
        # Create warped time axis
        original_time = np.linspace(0, 1, time_steps)
        warped_time = np.linspace(0, 1, new_time_steps)
        
        # Interpolate
        f = interp1d(original_time, ecg_signal, kind='linear')
        warped_signal = f(warped_time)
        
        # Resize back to original length
        if len(warped_signal) > time_steps:
            warped_signal = warped_signal[:time_steps]
        else:
            # Pad with zeros if shorter
            padding = time_steps - len(warped_signal)
            warped_signal = np.pad(warped_signal, (0, padding), 'constant')
        
        return warped_signal
    
    elif augmentation_type == 'amplitude_scale':
        # Amplitude scaling
        scale_factor = random.uniform(0.8, 1.2)
        return ecg_signal * scale_factor
    
    elif augmentation_type == 'time_shift':
        # Time shifting
        shift = random.randint(-50, 50)
        shifted_signal = np.roll(ecg_signal, shift)
        return shifted_signal
    
    elif augmentation_type == 'frequency_shift':
        # Frequency domain augmentation
        fft_signal = np.fft.fft(ecg_signal)
        # Apply random phase shift
        phase_shift = np.random.uniform(0, 2*np.pi, len(fft_signal))
        fft_signal *= np.exp(1j * phase_shift)
        return np.real(np.fft.ifft(fft_signal))
    
    return ecg_signal

def augment_minority_classes(segments, labels, target_counts):
    """Augment minority classes to reach target counts"""
    
    augmented_segments = []
    augmented_labels = []
    
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        current_count = len(label_indices)
        target_count = target_counts.get(label, current_count)
        
        print(f"Class {label}: {current_count} -> {target_count} samples")
        
        # Add original samples
        augmented_segments.extend(segments[label_indices])
        augmented_labels.extend([label] * current_count)
        
        # Generate augmented samples if needed
        if target_count > current_count:
            needed_samples = target_count - current_count
            augmentation_types = ['noise', 'time_warp', 'amplitude_scale', 'time_shift', 'frequency_shift']
            
            for i in range(needed_samples):
                # Randomly select original sample
                original_idx = np.random.choice(label_indices)
                original_signal = segments[original_idx]
                
                # Apply random augmentation
                aug_type = np.random.choice(augmentation_types)
                augmented_signal = augment_ecg_signal(original_signal, aug_type)
                
                augmented_segments.append(augmented_signal)
                augmented_labels.append(label)
    
    return np.array(augmented_segments), np.array(augmented_labels)
'''

    with open('augment_data.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    print("✅ Created augment_data.py script")


def create_synthetic_data_script():
    """Create script for synthetic data generation"""

    script_content = '''#!/usr/bin/env python3
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
'''

    with open('generate_synthetic.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    print("✅ Created generate_synthetic.py script")


def main():
    """Main CLI to manage downloads and helper script generation"""

    parser = argparse.ArgumentParser(description="Download additional ECG datasets (resume-friendly)")
    parser.add_argument("--records", type=str, default="", help="Comma-separated list of MIT-BIH record IDs to download (e.g., 200,201,202). If empty, use default set.")
    parser.add_argument("--no-ssl-verify", action="store_true", help="Disable SSL verification (use only if necessary)")
    parser.add_argument("--skip-others", action="store_true", help="Only process MIT-BIH downloads and skip helper script generation")
    args = parser.parse_args()

    print("📥 ECG Data Collection for Minority Classes")
    print("="*60)

    # Create data directories
    Path("data/raw/additional_mit_bih").mkdir(parents=True, exist_ok=True)
    Path("data/raw/ptb_xl").mkdir(parents=True, exist_ok=True)
    Path("data/raw/chapman").mkdir(parents=True, exist_ok=True)
    Path("data/raw/georgia").mkdir(parents=True, exist_ok=True)

    # Parse records list if provided
    records_list = None
    if args.records.strip():
        records_list = [r.strip() for r in args.records.split(",") if r.strip()]
        print(f"🎯 Restricting to records: {records_list}")

    # Download additional MIT-BIH records (resume-friendly)
    download_mit_bih_additional(records=records_list, verify_ssl=not args.no_ssl_verify)

    if args.skip_others:
        print("⏭️ Skipping augmentation/synthetic helper generation as requested.")
        return

    # Information about other datasets
    download_ptb_xl()
    download_chapman()
    download_georgia()

    # Create augmentation scripts (UTF-8 safe)
    create_data_augmentation_script()
    create_synthetic_data_script()

    print("\n📋 SUMMARY OF SOLUTIONS:")
    print("1. 📥 MIT-BIH: only missing files downloaded (resume-friendly)")
    print("2. 🔄 Data augmentation script created (augment_data.py)")
    print("3. 🧬 Synthetic data generation script created (generate_synthetic.py)")
    print("4. 📚 Manual downloads needed for large datasets")

    print("\n💡 NEXT STEPS:")
    print("1. Run: python augment_data.py (to augment existing data)")
    print("2. Run: python generate_synthetic.py (to create synthetic data)")
    print("3. Download PTB-XL, Chapman, or Georgia datasets manually")
    print("4. Combine all data sources for balanced training")

if __name__ == "__main__":
    main()
