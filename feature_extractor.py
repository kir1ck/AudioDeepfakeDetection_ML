import os
import librosa
import pandas as pd
import numpy as np
import scipy.stats

def get_stats(name, values, features):
    """Вспомогательная функция для сбора расширенной статистики."""
    # Убеждаемся, что работаем с одномерным массивом
    v = values.flatten()
    features[f'{name}_mean'] = np.mean(v)
    features[f'{name}_std'] = np.std(v)
    features[f'{name}_skew'] = scipy.stats.skew(v)
    features[f'{name}_kurt'] = scipy.stats.kurtosis(v)

def extract_features(file_path): # извлечение признаков из аудиофайла

    try:
        y, sr = librosa.load(file_path, sr=44100, mono=True)

        # duration, rms, zcr
        duration = librosa.get_duration(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)

        # spectral_centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_flatness = librosa.feature.spectral_flatness(y=y)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)


        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        features = {}

        get_stats('rms', rms, features)
        get_stats('zcr', zcr, features)
        get_stats('spec_cent', spec_cent, features)
        get_stats('spec_rolloff', spec_rolloff, features)
        get_stats('spec_flatness', spec_flatness, features)

        for i in range(13):
            get_stats(f'mfcc_{i}', mfcc[i], features)
            get_stats(f'mfcc_delta_{i}', mfcc_delta[i], features)
            get_stats(f'mfcc_delta2_{i}', mfcc_delta2[i], features)
            
        # Статистика для Spectral Contrast
        for i, block in enumerate(spec_contrast):
            get_stats(f'contrast_{i}', block, features)

        # Технические поля
        features['duration'] = duration
        features['filename'] = os.path.basename(file_path)

        return features
    
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return None
    

def process_audio_folder(folder_path, label=None): # перебор аудифоайлов по указанному пути и извлечение признаков

    all_features = []
    # Кэшируем список файлов
    file_list = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
    total_files = len(file_list)

    for i, file_name in enumerate(file_list):
        file_path = os.path.join(folder_path, file_name)
        if (i + 1) % 100 == 0: # Выводим прогресс каждые 100 файлов
            print(f'Обработка: {i + 1} из {total_files}')
            
        features = extract_features(file_path)

        if features:
            if label is not None:
                features['label'] = label
            all_features.append(features)

    return pd.DataFrame(all_features)



# перевод в DataFrame
df_training_real = process_audio_folder('D:/for-original/for-original/training/real', label=0)
df_training_fake = process_audio_folder('D:/for-original/for-original/training/fake', label=1)

df_training = pd.concat([df_training_real, df_training_fake], ignore_index=True)


df_testing_real = process_audio_folder('D:/for-original/for-original/testing/real', label=0)
df_testing_fake = process_audio_folder('D:/for-original/for-original/testing/fake', label=1)

df_testing = pd.concat([df_testing_real, df_testing_fake], ignore_index=True)


df_validation_real = process_audio_folder('D:/for-original/for-original/validation/real', label=0)
df_validation_fake = process_audio_folder('D:/for-original/for-original/validation/fake', label=1)

df_validation = pd.concat([df_validation_real, df_validation_fake], ignore_index=True)

# Вывод csv
df_training.to_csv('data2/for-original_train.csv')

df_testing.to_csv('data2/for-original_test.csv')

df_validation.to_csv('data2/for-original_validation.csv')


