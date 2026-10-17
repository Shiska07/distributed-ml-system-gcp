
import os
from Argparse import ArgumentParser

import mlflow
import numpy as np
import taxi_features
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from sklearn.metrics import (
    average_precision_score, f1_score, log_loss, precision_recall_curve, roc_auc_score, precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from training.models import get_model
from load_dataset import load_data, split_xy

MODEL_NAME = "taxi-tip-classifier"  

def pick_threshold(y, proba):
    """Probability cut-off that maximizes F1 (chosen on the validation set)."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    
    # get the threshold that maximizes the F1 score
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

def main():
    # -------------------------------------------------------------------------
    # 1. Arguments
    # -------------------------------------------------------------------------
    parser = ArgumentParser(description="Train a model on the taxi dataset.")
    parser.add_argument("--lake", type=str, required=True, help="Path to the data lake.")
    parser.add_argument("--version", type=str, required=True, help="Version of the dataset to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--cv_frac", type=float, default=0.2, help="Fraction of training data to use for cross-validation.")
    parser.add_argument("--model", type=str, required=True, help="Model tpe: LR, RF, XGB, etc.")  
    args = parser.parse_args()
    
    mlflow.set_experiment(args.experiment) 
    

    # 2. Data
    # -------------------------------------------------------------------------
    # load data
    splits, manifest = load_data(args.lake, args.version, args.seed)
    X_val, y_val = split_xy(splits["val"])
    X_test, y_test = split_xy(splits["test"])
    
    with mlflow.start_run(run_name=f"{args.model}_{args.version}_{args.seed}") as run:
        
        # 3. Run metadata: what is trained, on which data, from which code
        # ---------------------------------------------------------------------
        # model trained, data involved, and code versioning
        mlflow.set_tags({
            "model_type": args.model,
            "dataset_version": args.version,
            "dataset_source": manifest.get('source', 'unknown'),
            "Git_commit": args.git_commit if hasattr(args, 'git_commit') else 'unknown',
        })      

        mlflow.log_params({
            "seed": args.seed,
            "cv_frac": args.cv_frac,
            "train_size": len(splits["train"]),
            "val_size": len(X_val),
            "test_size": len(X_test),
        })
        
        mlflow.log_dict(manifest, "dataset/manifest.json")      # full lineage of the dataset used for training
        

        # 4. Hyperparameter tuning (sample of train, time-ordered CV)
        # ---------------------------------------------------------------------
        # get model and parameter grid
        model, param_grid = get_model(args.model)

        # get sample for cross validation
        cv_train = (splits["train"].sample(frac=args.cv_frac, random_state=args.seed)).sort_values(by="trip_start_ts")  
        
        # Set up GridSearchCV on a certain fraction of the training data for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=3),
                                   scoring="average_precision", n_jobs=-1, refit=False)
    
        # Fit the model
        X_cv, y_cv = split_xy(cv_train)
        grid_search.fit(X_cv, y_cv)
        
        # extract the best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # log best parameters and score to MLflow
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})                        
        mlflow.log_metric("cv_pr_auc", grid_search.best_score_)                                    
        mlflow.log_param("param_grid", str(param_grid)) 
    

        # 5. Final model: refit on the full training set
        # ---------------------------------------------------------------------
        # train the final model on the full training set with the best parameters
        final_model, _ = get_model(args.model, **best_params)
        X_train, y_train = split_xy(splits["train"])
        final_model.fit(X_train, y_train)
        

        # 6. Threshold (chosen on val) and evaluation (val + test)
        # ---------------------------------------------------------------------
        val_proba = final_model.predict_proba(X_val)[:, 1]
        test_proba = final_model.predict_proba(X_test)[:, 1]
        thresh = pick_threshold(y_val, val_proba)
        mlflow.log_param("threshold", thresh)
        
        val_metrics = evaluate(y_val, val_proba, thresh)
        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", float(value))

        test_metrics = evaluate(y_test, test_proba, thresh)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", float(value))
        

        # 7. Save the model to MLflow (pipeline + input schema + encoding code)
        # ---------------------------------------------------------------------
        # log model signature and save the model to MLflow
        example = X_val.head(100)
        flt32_cols = example.select_dtypes("float32").columns
        example = example.astype({col: "float64" for col in flt32_cols})  # convert float32 to float64 for MLflow signature
        model_info = mlflow.sklearn.log_model(
            model=final_model,
            model_name="model",
            example=example.head(5),
            signature=infer_signature(example, val_proba[:100]),
            code_paths=[os.path.dirname(taxi_features.__file__)]
        )
        

        # 8. Register the model and assign its role (@champion / @challenger)
        # ---------------------------------------------------------------------
        # regiser the model in the MLflow Model Registry
        client = MlflowClient()
        version = mlflow.register_model(model_info.model_url, MODEL_NAME).version
        client.set_model_version_tag(MODEL_NAME, version, "threshold", str(thresh))
        client.set_model_version_tag(MODEL_NAME, version, "model_type", str(args.model))
        try:
            # if a champion model exists already, set alias to challenger, otherwise set alias to champion
            client.get_model_version_by_alias(MODEL_NAME, "champion")
            alias = "challenger"
        except MlflowException:
            alias = "champion"
        client.set_registered_model_alias(MODEL_NAME, version, alias)
        
        print(f"Model registered as {MODEL_NAME} version {version} with alias '{alias}'")
        
    

if __name__ == "__main__":
    main()
    