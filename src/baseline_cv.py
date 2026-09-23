"""
baseline_cv.py
--------------
Stratifizierte 5-fold Cross-Validation der klassischen Baseline auf den
Trainingsdaten. Liefert Mittelwert und Standardabweichung der Metriken für
die Clickbait-Klasse und prüft damit die Stabilität der Baseline gegenüber
unterschiedlichen Datenaufteilungen.

Das Testset wird hier nicht verwendet.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from baseline import build_pipeline

SEED = 42
N_FOLDS = 5


def run_cv(train_df: pd.DataFrame, model: str) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_validate(
        build_pipeline(model),
        train_df["headline"],
        train_df["label"],
        cv=cv,
        scoring=["f1", "precision", "recall", "accuracy"],
    )
    return pd.DataFrame({
        "fold": range(1, N_FOLDS + 1),
        "f1": scores["test_f1"],
        "precision": scores["test_precision"],
        "recall": scores["test_recall"],
        "accuracy": scores["test_accuracy"],
    })


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    train_df = pd.read_csv(root / "data" / "train.csv")
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)

    summary_rows = []
    for model in ["logreg", "svm"]:
        folds = run_cv(train_df, model)
        folds.insert(0, "model", model)
        folds.to_csv(out_dir / f"baseline_cv_{model}.csv", index=False)

        print(f"\n===== {N_FOLDS}-fold CV: {model} =====")
        print(folds.drop(columns="model").round(4).to_string(index=False))

        row = {"model": model}
        for m in ["f1", "precision", "recall", "accuracy"]:
            row[f"{m}_mean"] = folds[m].mean()
            row[f"{m}_std"] = folds[m].std()
        summary_rows.append(row)
        print(f"F1 = {row['f1_mean']:.3f} ± {row['f1_std']:.3f}")

    summary = pd.DataFrame(summary_rows).round(4)
    summary.to_csv(out_dir / "baseline_cv_summary.csv", index=False)
    print("\nGespeichert: results/baseline_cv_summary.csv")
