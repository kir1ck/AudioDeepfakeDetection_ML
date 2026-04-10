
def create_objective(X_tr, X_valid, y_tr, y_valid):

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 5, 100)
        max_depth = trial.suggest_int('max_depth', 1, 25)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

        clf = ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            criterion='entropy',
            max_depth=max_depth,
            max_features='sqrt',
            min_samples_leaf=min_samples_leaf
        )
        clf.fit(X_tr, y_tr)

        y_score = clf.predict_proba(X_valid)[:, 1]

        eer, best_threshold = caluculate_eer(y_valid, y_score)

        trial.set_user_attr("eer_threshold", float(best_threshold))

        trial.set_user_attr("accuracy", metrics.accuracy_score(y_valid, y_score > 0.5))


        # y_pred_valid = clf.predict(X_test)
        
        # score = metrics.accuracy_score(y_test, y_pred_test)

        return eer
    return objective

