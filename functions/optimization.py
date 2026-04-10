db_url = "sqlite:///XGB_research.db"
for name, (X_tr, X_valid, y_tr, y_valid) in datasets.items():
    print(f"\n>>> Запуск оптимизации для датасета: {name}")

    # уникальное имя исследования
    study_name = f"xgb_{name}"

    try:
        optuna.delete_study(study_name=study_name, storage=db_url)
    except KeyError:
        pass

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=db_url,
        load_if_exists=True,
        
    )

    obj_func = create_objective(X_tr, X_valid, y_tr, y_valid)

    study.optimize(obj_func, n_trials=100)

    print(f"Лучшие параметры для {name}: {study.best_params} с показателем EER: {study.best_value}")