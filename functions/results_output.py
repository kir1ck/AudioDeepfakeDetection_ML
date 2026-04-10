final_metrics = []

custom_thresholds = {
      'original': 0.25,
      'norm': 0.31,
      '2sec': 0.21,
      'rerec': 0.45, 
}

fig, axes = plt.subplots(4,2, figsize=(15,12))
# axes = axes.ravel()

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
    y_proba_pred = model.predict_proba(X_test)[:, 1]

    eer, _ = caluculate_eer(y_test, y_proba_pred)

    y_proba_pred = pd.Series(y_proba_pred)

    current_thresh = custom_thresholds[name]

    y_pred_opt = y_proba_pred.apply(lambda x: 1 if x > current_thresh else 0)
    
    # Расчет метрик
    acc = metrics.accuracy_score(y_test, y_pred_opt)
    f1 = metrics.f1_score(y_test, y_pred_opt)
    precision = metrics.precision_score(y_test, y_pred_opt)
    recall = metrics.recall_score(y_test, y_pred_opt)
    
    # Сохраняем метрики в список
    final_metrics.append({
        'Dataset': name,
        'EER': eer,
        'Accuracy': acc,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall
    })
    
    # Визуализация Confusion Matrix
    cm = metrics.confusion_matrix(y_test, y_pred_opt)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i, 0], cmap='Blues', cbar=False)
    axes[i, 0].set_title(f'Confusion Matrix: {name.upper()}')
    axes[i, 0].set_xlabel('Predicted labels')
    axes[i, 0].set_ylabel('True labels')

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances.nlargest(10).sort_values().plot(kind='barh', ax=axes[i, 1], color='teal')
    axes[i, 1].set_title(f"Top 10 Features (Artifacts): {name.upper()}", fontsize=12)
    axes[i, 1].set_xlabel('Importance Score')

# Корректировка расположения графиков
plt.tight_layout()
plt.show()

# 4. Вывод итоговой сводной таблицы
df_results = pd.DataFrame(final_metrics)
df_results.to_excel('results/research1_rf_result.xlsx')
print("\n--- Итоговые результаты по всем датасетам ---")
print(df_results)