"""Train the tip classifier and log everything to MLflow (and optionally Vertex AI Experiments)."""
import os
from argparse import ArgumentParser

import mlflow
import numpy as np
import taxi_features
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.metrics import (
    average_precision_score, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import tracking
from load_dataset import load_data, split_xy
from models import get_model

MODEL_NAME = "taxi-tip-classifier"


# Evaluation helpers
# ------------------------------------------------------------------------------

def pick_threshold(y, proba):
    """Probability cut-off that maximizes F1 (chosen on the validation set)."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_thresh = thresholds[np.argmax(f1[:-1])]
    return float(best_thresh)


def evaluate(y, proba, threshold):
    """Ranking metrics (from probabilities) + classification metrics (at the threshold)."""
    pred = (proba >= threshold).astype(int)
    return {
        # ranking quality (threshold-independent)
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "log_loss": log_loss(y, proba),
        # yes/no decisions at the chosen threshold
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
    }


# Training run
# ------------------------------------------------------------------------------

def main():
    # 1. Arguments
    # -------------------------------------------------------------------------
    parser = ArgumentParser(description="Train a model on the taxi dataset.")
    parser.add_argument("--lake", type=str, required=True, help="Path to the data lake.")
    parser.add_argument("--version", type=str, required=True, help="Version of the dataset to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--cv_frac", type=float, default=0.2, help="Fraction of training data to use for cross-validation.")
    parser.add_argument("--model", type=str, required=True, choices=["LR", "RF", "XGB"], help="Model type: LR, RF, XGB.")
    parser.add_argument("--experiment", type=str, default="taxi-tips", help="MLflow experiment name.")
    parser.add_argument("--git_commit", type=str, default=os.environ.get("GIT_COMMIT", "unknown"),
                        help="Code version, logged with the run.")
    parser.add_argument("--vertex_experiment", type=str, default="",
                        help="Vertex AI experiment name; empty = MLflow only.")
    args = parser.parse_args()

    tracking.refresh_mlflow_token()
    tracking.init_vertex(args.vertex_experiment)
    mlflow.set_experiment(args.experiment)

    # 2. Data
    # -------------------------------------------------------------------------
    splits, manifest = load_data(args.lake, args.version, args.seed)
    X_val, y_val = split_xy(splits["val"])
    X_test, y_test = split_xy(splits["test"])

    run_name = f"{args.model}_{args.version}_{args.seed}"
    with mlflow.start_run(run_name=run_name) as run:
        tracking.start_vertex_run(run_name, run.info.run_id)

        # 3. Run metadata: what is trained, on which data, from which code
        # ---------------------------------------------------------------------
        mlflow.set_tags({
            "model_type": args.model,
            "dataset_version": args.version,
            "dataset_source": manifest.get("source", "unknown"),
            "git_commit": args.git_commit,
        })
        tracking.log_params({
            "seed": args.seed,
            "cv_frac": args.cv_frac,
            "train_size": len(splits["train"]),
            "val_size": len(X_val),
            "test_size": len(X_test),
        })
        mlflow.log_dict(manifest, "dataset/manifest.json")      # full lineage of the dataset

        # 4. Hyperparameter tuning (sample of train, time-ordered CV)
        # ---------------------------------------------------------------------
        model, param_grid = get_model(args.model)
        cv_train = (splits["train"]
                    .sample(frac=args.cv_frac, random_state=args.seed)
                    .sort_values(by="trip_start_ts"))
        grid_search = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=3),
                                   scoring="average_precision", n_jobs=-1, refit=False)
        X_cv, y_cv = split_xy(cv_train)
        grid_search.fit(X_cv, y_cv)

        best_params = grid_search.best_params_
        tracking.log_params({f"best_{k}": v for k, v in best_params.items()})
        tracking.log_params({"param_grid": str(param_grid)})
        tracking.log_metrics({"cv_pr_auc": grid_search.best_score_})

        # 5. Final model: refit on the full training set
        # ---------------------------------------------------------------------
        final_model, _ = get_model(args.model)
        final_model.set_params(**best_params)
        X_train, y_train = split_xy(splits["train"])
        final_model.fit(X_train, y_train)

        # 6. Threshold (chosen on val) and evaluation (val + test)
        # ---------------------------------------------------------------------
        val_proba = final_model.predict_proba(X_val)[:, 1]
        test_proba = final_model.predict_proba(X_test)[:, 1]
        thresh = pick_threshold(y_val, val_proba)
        tracking.log_params({"threshold": thresh})

        val_metrics = evaluate(y_val, val_proba, thresh)
        test_metrics = evaluate(y_test, test_proba, thresh)
        tracking.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        tracking.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # 7. Save the model to MLflow (pipeline + input schema + encoding code)
        # ---------------------------------------------------------------------
        tracking.refresh_mlflow_token()          # long runs: token may have expired
        example = X_val.head(100)
        flt32_cols = example.select_dtypes("float32").columns
        example = example.astype({col: "float64" for col in flt32_cols})  # JSON numbers arrive as float64
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            input_example=example.head(5),
            signature=infer_signature(example, val_proba[:100]),
            code_paths=[os.path.dirname(taxi_features.__file__)],
        )

        # 8. Register the model and assign its role (@champion / @challenger)
        # ---------------------------------------------------------------------
        client = MlflowClient()
        version = mlflow.register_model(model_info.model_uri, MODEL_NAME).version
        client.set_model_version_tag(MODEL_NAME, version, "threshold", str(thresh))
        client.set_model_version_tag(MODEL_NAME, version, "model_type", str(args.model))
        try:
            # a champion exists already -> new model is the challenger
            client.get_model_version_by_alias(MODEL_NAME, "champion")
            alias = "challenger"
        except MlflowException:
            alias = "champion"
        client.set_registered_model_alias(MODEL_NAME, alias, version)

        tracking.end_vertex_run()
        print(f"Model registered as {MODEL_NAME} version {version} with alias '{alias}'")


if __name__ == "__main__":
    main()
    