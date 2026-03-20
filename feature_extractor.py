import os
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_features(file_path): # извлечение признаков из аудиофайла

    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)

        # duration
        duration = librosa.get_duration(y=y, sr=sr)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # spectral_centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)
        spec_cent_std = np.std(spec_cent)

        # rolloff
        roloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(roloff)

        # zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        zcr_mean = np.mean(zcr)

        # rms 
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        # spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # tonnetz
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)

        features = {}

        for i, (m, s) in enumerate(zip(mfcc_mean, mfcc_std)):
            features[f'mfcc_{i}_mean'] = m
            features[f'mfcc_{i}_std'] = s

        for i, val in enumerate(chroma_mean):
            features[f'chroma_{i}'] = val
        
        features['spec_cent_mean'] = spec_cent_mean
        features['spec_cent_std'] = spec_cent_std
        features['roloff_mean'] = rolloff_mean
        features['zcr_mean'] = zcr_mean
        features['rms_mean'] = rms_mean

        for i, val in enumerate(contrast_mean):
            features[f'contrast_{i}'] = val

        for i, val in enumerate(tonnetz_mean):
            features[f'tonnetz_{i}'] = val
        
        features['duration'] = duration
        features['filename'] = os.path.basename(file_path)

        return features
    
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return None
    

def process_audio_folder(folder_path, label=None): # перебор аудифоайлов по указанному пути и извлечение признаков

    all_features = []
    counter = 1

    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.wav', '.mp3', '.flac', '.m4a')):
            file_path = os.path.join(folder_path, file_name)
            counter += 1
            features = extract_features(file_path)

            if features:
                if label is not None:
                    features['label'] = label
                all_features.append(features)

    return pd.DataFrame(all_features)



# перевод в DataFrame
df_training_real = process_audio_folder('D:/for-rerec/for-rerecorded/training/real', label=0)
df_training_fake = process_audio_folder('D:/for-rerec/for-rerecorded/training/fake', label=1)

df_training = pd.concat([df_training_real, df_training_fake], ignore_index=True)


df_testing_real = process_audio_folder('D:/for-rerec/for-rerecorded/testing/real', label=0)
df_testing_fake = process_audio_folder('D:/for-rerec/for-rerecorded/testing/fake', label=1)

df_testing = pd.concat([df_testing_real, df_testing_fake], ignore_index=True)


df_validation_real = process_audio_folder('D:/for-rerec/for-rerecorded/validation/real', label=0)
df_validation_fake = process_audio_folder('D:/for-rerec/for-rerecorded/validation/fake', label=1)

df_validation = pd.concat([df_validation_real, df_validation_fake], ignore_index=True)

# Вывод csv
df_training.to_csv('data/for-rerec_training.csv')

df_testing.to_csv('data/for-rerec_testing.csv')

df_validation.to_csv('data/for-rerec_validation.csv')


