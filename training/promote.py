"""Champion / challenger promotion.

Compares the @challenger model with the @champion on their logged test metrics and
moves @champion to the challenger only if it wins by the rule below. Both must be
trained on the same dataset version, so their test splits are identical.

Rule: challenger test PR-AUC >= champion + MIN_GAIN, and test log loss not worse.

Usage:
    python promote.py                 # compare and promote if the rule passes
    python promote.py --dry-run       # compare and print the decision only
"""
from argparse import ArgumentParser

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

import tracking
from train import MODEL_NAME


def version_summary(client: MlflowClient, alias: str) -> dict:
    """Registry version behind an alias, plus the metrics of the run that trained it."""
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    run = client.get_run(mv.run_id)
    return {
        "alias": alias,
        "version": mv.version,
        "model_type": run.data.tags.get("model_type", "?"),
        "dataset_version": run.data.tags.get("dataset_version", "?"),
        "pr_auc": run.data.metrics["test_pr_auc"],
        "log_loss": run.data.metrics["test_log_loss"],
    }


def main():
    parser = ArgumentParser(description="Promote the challenger if it beats the champion.")
    parser.add_argument("--min_gain", type=float, default=0.005,
                        help="Minimum PR-AUC improvement required to promote.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the decision without changing any alias.")
    args = parser.parse_args()

    tracking.refresh_mlflow_token()
    client = MlflowClient()

    # 1. Champion and challenger -------------------------------------------------
    try:
        champ = version_summary(client, "champion")
        chall = version_summary(client, "challenger")
    except MlflowException as e:
        raise SystemExit(f"Need both @champion and @challenger: {e}")

    for m in (champ, chall):
        print(f"@{m['alias']:<10} v{m['version']} {m['model_type']:<4} "
              f"data={m['dataset_version']}  PR-AUC={m['pr_auc']:.4f}  log loss={m['log_loss']:.4f}")

    # 2. Same test set? ------------------------------------------------------------
    if champ["dataset_version"] != chall["dataset_version"]:
        raise SystemExit("Different dataset versions: test sets differ, metrics are not comparable.")

    # 3. Rule -----------------------------------------------------------------------
    gain = chall["pr_auc"] - champ["pr_auc"]
    promote = gain >= args.min_gain and chall["log_loss"] <= champ["log_loss"]
    reason = (f"PR-AUC gain {gain:+.4f} (need >= {args.min_gain}), "
              f"log loss {chall['log_loss']:.4f} vs {champ['log_loss']:.4f}")
    print(("PROMOTE: " if promote else "KEEP champion: ") + reason)

    if args.dry_run:
        return

    # 4. Apply ----------------------------------------------------------------------
    if promote:
        client.set_registered_model_alias(MODEL_NAME, "previous_champion", champ["version"])
        
if __name__ == "__main__":
    main()
        