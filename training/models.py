""" Model factory for creating different types of classifiers. """
import params_cv
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_model(model_type, **kwargs):
    """Return a model instance based on the specified type."""
    if model_type == "LR":
        return LogisticRegression(**kwargs), params_cv.params_lr
    elif model_type == "RF":
        return RandomForestClassifier(**kwargs), params_cv.params_rfc
    elif model_type == "XGB":
        return XGBClassifier(**kwargs), params_cv.params_xgb
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    