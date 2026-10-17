""" Parameter grids for hyperparameter tuning. """

# LogisticRegression(max_iter=1000)                                -> 4 x 2 = 8 combinations
params_lr = {
    "model__C": [0.01, 0.1, 1.0, 10.0],             # inverse regularization strength
    "model__class_weight": [None, "balanced"],      # 70/30 imbalance: weight the minority class or not
}

# RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=seed)
#                                                                  -> 2 x 2 x 2 = 8 combinations
params_rfc = {
    "model__max_depth": [10, 20],                   # tree depth: complexity vs overfitting
    "model__min_samples_leaf": [50, 200],           # min rows per leaf: smooths noisy splits on large data
    "model__class_weight": [None, "balanced_subsample"],
}

# XGBClassifier(n_estimators=300, subsample=0.8, colsample_bytree=0.8,
#               tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=seed)
#                                                                  -> 3 x 2 x 2 = 12 combinations
params_xgb = {
    "model__max_depth": [4, 6, 8],                  # tree depth
    "model__learning_rate": [0.05, 0.1],            # step size per tree
    "model__scale_pos_weight": [1.0, 2.3],          # ~ negatives / positives (70/30); XGBoost's class_weight
}
