""" Model factory for creating different types of classifiers. """
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from params_cv import params_lr, params_rfc, params_xgb
from taxi_features.encoding import build_encoder_lr, build_encoder_trees

SEED=42

MODELS ={
    "LR": (build_encoder_lr, LogisticRegression(max_iter=1000), params_lr),
    "RF": (build_encoder_trees, RandomForestClassifier(n_estimators=200, max_features="sqrt",
                               n_jobs=-1,random_state=SEED), params_rfc),
    "XGB": (build_encoder_trees, XGBClassifier(n_estimators=300, random_state=SEED), params_xgb),
}

def get_model(name: str):
    """Return (unfitted pipeline, parameter grid) for a model type."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from {list(MODELS)}")
    
    build_encoder, model, param_grid = MODELS[name]
    pipeline = Pipeline([
        ("encoder", build_encoder()),
        ("model", clone(model))
    ])
    return pipeline, param_grid
