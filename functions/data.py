training_df = pd.read_csv('data/training.csv')
testing_df = pd.read_csv('data/testing.csv')
validation_df = pd.read_csv('data/validation.csv')

# векторы признаков
X_train = training_df.drop(['Unnamed: 0','filename', 'label', 'duration'], axis=1)
X_test = testing_df.drop(['Unnamed: 0','filename', 'label', 'duration'], axis=1)
X_valid = validation_df.drop(['Unnamed: 0','filename', 'label', 'duration'], axis=1)

X_train_valid = pd.concat([X_train, X_valid], ignore_index=False)

# векторы ответов
y_train = training_df['label']
y_test = testing_df['label']
y_valid = validation_df['label']

y_train_valid = pd.concat([y_train, y_valid], ignore_index=False)

norm_training_df = pd.read_csv('data/for-norm_training.csv')
norm_testing_df = pd.read_csv('data/for-norm_testing.csv')
norm_validation_df = pd.read_csv('data/for-norm_validation.csv')

X_norm_training = norm_training_df.drop(['Unnamed: 0', 'filename', 'duration', 'label'], axis=1)
X_norm_testing = norm_testing_df.drop(['Unnamed: 0', 'filename', 'duration', 'label'], axis=1)
X_norm_validation = norm_validation_df.drop(['Unnamed: 0', 'filename', 'duration', 'label'], axis=1)

X_norm_train_valid = pd.concat([X_norm_training, X_norm_validation], ignore_index=False)

y_norm_training = norm_training_df['label']
y_norm_testing = norm_testing_df['label']
y_norm_validation = norm_validation_df['label']

y_norm_train_valid = pd.concat([y_norm_training, y_norm_validation], ignore_index=False)


sec_train = pd.read_csv('data/for-2sec_training.csv')
sec_test = pd.read_csv('data/for-2sec_testing.csv')
sec_valid = pd.read_csv('data/for-2sec_validation.csv')


X_sec_train = sec_train.drop(['Unnamed: 0', 'filename', 'duration', 'label'], axis=1)
X_sec_test = sec_test.drop(['Unnamed: 0', 'filename', 'duration', 'label'], axis=1)
X_sec_valid = sec_valid.drop(['Unnamed: 0', 'filename', 'duration', 'label'], axis=1)

X_sec_train_valid = pd.concat([X_sec_train, X_sec_valid], ignore_index=False)

y_sec_train = sec_train['label']
y_sec_test = sec_test['label']
y_sec_valid = sec_valid['label']

y_sec_train_valid = pd.concat([y_sec_train, y_sec_valid], ignore_index=False)

rerec_train_df = pd.read_csv('data/for-rerec_training.csv')
rerec_test_df = pd.read_csv('data/for-rerec_testing.csv')
rerec_valid_df = pd.read_csv('data/for-rerec_validation.csv')

X_rerec_train = rerec_train_df.drop(['Unnamed: 0', 'duration', 'filename', 'label'], axis=1)
X_rerec_test = rerec_test_df.drop(['Unnamed: 0', 'duration', 'filename', 'label'], axis=1)
X_rerec_valid = rerec_valid_df.drop(['Unnamed: 0', 'duration', 'filename', 'label'], axis=1)

X_rerec_train_valid = pd.concat([X_rerec_train, X_rerec_valid], ignore_index=False)

y_rerec_train = rerec_train_df['label']
y_rerec_test = rerec_test_df['label']
y_rerec_valid = rerec_valid_df['label']

y_rerec_train_valid = pd.concat([y_rerec_train, y_rerec_valid], ignore_index=False)


datasets = {
    'original': {'train': (X_train_valid, y_train_valid) , 'test': (X_test, y_test)},
    'norm': {'train': (X_norm_train_valid, y_norm_train_valid), 'test': (X_norm_testing, y_norm_testing)},
    '2sec': {'train': (X_sec_train_valid, y_sec_train_valid), 'test': (X_sec_test, y_sec_test) },
    'rerec': {'train': (X_rerec_train_valid, y_rerec_train_valid), 'test': (X_rerec_test, y_rerec_test)}
}


rf_best_params = {}
datasets_names = ['original', 'norm', 'sec', 'rerec']

for name in datasets_names:
    study = optuna.load_study(study_name=f'rf_{name}', storage=storage_url)
    rf_best_params[name] = study.best_params


xgb_best_params = {}
datasets_names = ['original', 'norm', 'sec', 'rerec']

for name in datasets_names:
    study = optuna.load_study(study_name=f'xgb_{name}', storage='sqlite:///XGB_research3.db')
    xgb_best_params[name] = study.best_params


knn_best_params = {}
datasets_names = ['original', 'norm', 'sec', 'rerec']

for name in datasets_names:
    study = optuna.load_study(study_name=f'knn_{name}', storage='sqlite:///knn_research3.db')
    knn_best_params[name] = study.best_params