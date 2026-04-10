final_metrics = []



thresholds = np.arange(0.1, 1, 0.05)

fig, axes = plt.subplots(4,2, figsize=(15,12))

for i, (name, data) in enumerate(datasets.items()):
    # Извлекаем данные
    X_train, y_train = data['train']
    X_test, y_test = data['test']
    
    # Извлекаем параметры именно для этого датасета
    params = rf_best_params[name]
    
    # Инициализация и обучение модели
    model = ensemble.RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Предсказание
    y_pred = model.predict_proba(X_test)[:,1]
    y_score = pd.Series(y_pred)

    recall_scores = []
    precision_scores = []
    f1_scores = []

    for threshold in thresholds:
            # Если вероятность > threshold, то 1, иначе 0
            y_pred_custom = (y_score > threshold).astype(int)
            
            recall_scores.append(metrics.recall_score(y_test, y_pred_custom))
            precision_scores.append(metrics.precision_score(y_test, y_pred_custom))
            f1_scores.append(metrics.f1_score(y_test, y_pred_custom))

    # 5. Визуализация на i-м графике
    ax = axes[i]
    ax.plot(thresholds, recall_scores, label='Recall', marker='.', alpha=0.7)
    ax.plot(thresholds, precision_scores, label='Precision', marker='.', alpha=0.7)
    ax.plot(thresholds, f1_scores, label='F1-score', lw=3, color='black') # Выделим F1 пожирнее

    # Оформление графика
    ax.set_title(f'Threshold Analysis: {name.upper()}', fontsize=14)
    ax.set_xlabel('Probability Threshold')
    ax.set_ylabel('Score')
    ax.set_xticks(np.arange(0.1, 1.05, 0.1)) # Сетка чуть пореже для читаемости
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()


plt.tight_layout()
plt.show()
    
    