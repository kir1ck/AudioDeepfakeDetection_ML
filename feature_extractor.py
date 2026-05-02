import os
import librosa
import pandas as pd
import numpy as np
import scipy.stats
import scipy.fftpack as fftpack

def get_stats(name, values, features, stats_list=None): # извлечение статистических показателей
    if stats_list is None:
        stats_list = ['mean', 'std', 'skew', 'kurtosis']
    
    v = values.flatten()
    if 'mean' in stats_list:
        features[f'{name}_mean'] = np.mean(v)
    if 'std' in stats_list:
        features[f'{name}_std'] = np.std(v)
    if 'skew' in stats_list:
        features[f'{name}_skew'] = scipy.stats.skew(v)
    if 'kurtosis' in stats_list:
        features[f'{name}_kurtosis'] = scipy.stats.kurtosis(v)

def extract_features(file_path, feature_groups=None, n_mfcc=20, n_lfcc=20, stats_list=None): # извлечение признаков из аудиофайла

    if feature_groups is None:
        feature_groups = ['rms', 'zcr', 'spec_cent', 'spec_rolloff', 'spec_flatness', 'mfcc', 'lfcc', 'contrast']
    if stats_list is None:
        stats_list = ['mean', 'std', 'skew', 'kurtosis']

    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # duration, rms, zcr
        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y) if 'rms' in feature_groups else None
        zcr = librosa.feature.zero_crossing_rate(y=y) if 'zcr' in feature_groups else None

        # spectral_centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr) if 'spec_cent' in feature_groups else None
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr) if 'spec_rolloff' in feature_groups else None
        spec_flatness = librosa.feature.spectral_flatness(y=y) if 'spec_flatness' in feature_groups else None
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr) if 'contrast' in feature_groups else None


        # MFCC
        if 'mfcc' in feature_groups:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        else:
            mfcc = mfcc_delta = mfcc_delta2 = None

        if 'lfcc' in feature_groups:
            S = np.abs(librosa.stft(y))
            lfcc = fftpack.dct(S, axis=0, type=2, norm='ortho')[:n_lfcc]
            lfcc_delta = librosa.feature.delta(lfcc)
            lfcc_delta2 = librosa.feature.delta(lfcc, order=2)
        else:
            lfcc = lfcc_delta = lfcc_delta2 = None

        features = {}

        if rms is not None:
            get_stats('rms', rms, features, stats_list)
        if zcr is not None:
            get_stats('zcr', zcr, features, stats_list)
        if spec_cent is not None:
            get_stats('spec_cent', spec_cent, features, stats_list)
        if spec_rolloff is not None:
            get_stats('spec_rolloff', spec_rolloff, features, stats_list)
        if spec_flatness is not None:
            get_stats('spec_flatness', spec_flatness, features, stats_list)

        if mfcc is not None:
            for i in range(n_mfcc):
                get_stats(f'mfcc_{i}', mfcc[i], features, stats_list)
                get_stats(f'mfcc_delta_{i}', mfcc_delta[i], features, stats_list)
                get_stats(f'mfcc_delta2_{i}', mfcc_delta2[i], features, stats_list)
        
        if lfcc is not None:
            for i in range(n_lfcc):
                get_stats(f'lfcc_{i}', lfcc[i], features, stats_list)
                get_stats(f'lfcc_delta_{i}', lfcc_delta[i], features, stats_list)
                get_stats(f'lfcc_delta2_{i}', lfcc_delta2[i], features, stats_list)
            
        if spec_contrast is not None:
            # Статистика для Spectral Contrast
            for i, block in enumerate(spec_contrast):
                get_stats(f'contrast_{i}', block, features, stats_list)

        # Технические поля
        features['duration'] = duration
        features['filename'] = os.path.basename(file_path)

        return features
    
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return None
    

def process_audio_folder(folder_path, feature_groups=None, n_mfcc=20, n_lfcc=20, stats_list=None, progress_callback=None, total_files=None, processed=0): # перебор аудифоайлов по указанному пути и извлечение признаков
    """Process files in a folder with 'real' and 'fake' subfolders."""
    all_features = []
    # Check if folder has subfolders for labels
    subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    if subfolders:
        # Assume subfolders are 'real' (0) and 'fake' (1)
        for subfolder in subfolders:
            sub_path = os.path.join(folder_path, subfolder)
            if 'real' in subfolder.lower():
                label = 0
            elif 'fake' in subfolder.lower():
                label = 1
            else:
                continue  # Skip unknown subfolders
            # Process files in subfolder
            file_list = [f for f in os.listdir(sub_path) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
            for file_name in file_list:
                file_path = os.path.join(sub_path, file_name)
                features = extract_features(file_path, feature_groups=feature_groups, n_mfcc=n_mfcc, n_lfcc=n_lfcc, stats_list=stats_list)

                if features:
                    features['label'] = label
                    all_features.append(features)
                processed += 1
                if progress_callback is not None and total_files:
                    progress_callback(processed / total_files * 100, f"Processed {processed} / {total_files} files")
    else:
        # No subfolders, process all files with default label or something, but since we assume structure, perhaps error
        raise ValueError("Dataset folder must contain subfolders for different classes (e.g., 'real' and 'fake')")

    return pd.DataFrame(all_features), processed


def process_dataset(dataset_root, feature_groups=None):
    """Process entire dataset with training/validation/testing splits.
    
    Expected structure:
    dataset_root/
        training/
            real/
            fake/
        validation/
            real/
            fake/
        testing/
            real/
            fake/
    
    Outputs 3 CSV files: training.csv, validation.csv, testing.csv
    Each file contains features from both real (label=0) and fake (label=1) samples.
    """
    splits = ['training', 'validation', 'testing']
    output_dir = dataset_root
    
    file_exts = ('.wav', '.mp3', '.flac', '.m4a')
    total_files = 0
    for split_path in [os.path.join(dataset_root, split) for split in splits if os.path.isdir(os.path.join(dataset_root, split))]:
        for subfolder in os.listdir(split_path):
            sub_path = os.path.join(split_path, subfolder)
            if not os.path.isdir(sub_path):
                continue
            total_files += len([f for f in os.listdir(sub_path) if f.lower().endswith(file_exts)])

    processed = 0
    for split in splits:
        split_path = os.path.join(dataset_root, split)
        if not os.path.isdir(split_path):
            print(f"Warning: {split_path} not found, skipping...")
            continue
        
        print(f"\nProcessing {split} split...")
        df, processed = process_audio_folder(
            split_path,
            feature_groups=feature_groups,
            total_files=total_files,
            processed=processed
        )
        
        if df.empty:
            print(f"Warning: No features extracted for {split}")
            continue
        
        output_path = os.path.join(output_dir, f'{split}.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved {split}.csv with {len(df)} samples")
        print(f"  - Real (label=0): {len(df[df['label']==0])} samples")
        print(f"  - Fake (label=1): {len(df[df['label']==1])} samples")


def process_dataset_paths(training_folder, validation_folder, testing_folder, feature_groups=None, n_mfcc=20, n_lfcc=20, stats_list=None, output_dir=None, progress_callback=None):
    """Process explicit split folders and save training/validation/testing CSVs."""
    import time
    extraction_start_time = time.time()
    
    if output_dir is None:
        try:
            common_parent = os.path.commonpath([training_folder, validation_folder, testing_folder])
            if os.path.isdir(common_parent):
                output_dir = common_parent
            else:
                output_dir = os.path.dirname(training_folder)
        except ValueError:
            output_dir = os.path.dirname(training_folder)
    
    # Create extraction_result folder with feature set name
    feature_set_name = '_'.join(sorted(feature_groups)) if feature_groups else 'features'
    extraction_result_dir = os.path.join(output_dir, 'extraction_result', feature_set_name)
    os.makedirs(extraction_result_dir, exist_ok=True)

    splits = {
        'training': training_folder,
        'validation': validation_folder,
        'testing': testing_folder
    }
    output_paths = {}

    # Count total files for progress
    file_exts = ('.wav', '.mp3', '.flac', '.m4a')
    total_files = 0
    for split_path in splits.values():
        if not os.path.isdir(split_path):
            continue
        for subfolder in os.listdir(split_path):
            sub_path = os.path.join(split_path, subfolder)
            if not os.path.isdir(sub_path):
                continue
            total_files += len([f for f in os.listdir(sub_path) if f.lower().endswith(file_exts)])

    processed = 0
    for split, split_path in splits.items():
        if not os.path.isdir(split_path):
            raise ValueError(f"Split folder not found: {split_path}")

        print(f"\nProcessing {split} split...")
        df, processed = process_audio_folder(
            split_path,
            feature_groups=feature_groups,
            n_mfcc=n_mfcc,
            n_lfcc=n_lfcc,
            stats_list=stats_list,
            progress_callback=progress_callback,
            total_files=total_files,
            processed=processed
        )

        if df.empty:
            print(f"Warning: No features extracted for {split}")
            continue

        output_path = os.path.join(extraction_result_dir, f'{split}.csv')
        df.to_csv(output_path, index=False)
        output_paths[split] = output_path

        print(f"Saved {split}.csv with {len(df)} samples to {extraction_result_dir}")
        print(f"  - Real (label=0): {len(df[df['label']==0])} samples")
        print(f"  - Fake (label=1): {len(df[df['label']==1])} samples")
    
    extraction_time = time.time() - extraction_start_time
    output_paths['extraction_time'] = extraction_time
    output_paths['feature_set_name'] = feature_set_name
    output_paths['extraction_result_dir'] = extraction_result_dir
    
    return output_paths


