def get_coef_from_shap_values(shap_values, X_train_scaled):
    w = shap_values.values * (X_train_scaled - X_train_scaled.mean(0))
    return w.mean(0)