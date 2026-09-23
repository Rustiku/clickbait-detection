"""
baseline.py
-----------
Klassische, merkmalsbasierte Baseline: TF-IDF + Logistic Regression / LinearSVM.
Dient als Vergleichsmaßstab, an dem der Mehrwert der Transformer-Modelle
gemessen wird.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def build_pipeline(model: str = "logreg") -> Pipeline:
    """
    Baut eine sklearn-Pipeline aus TF-IDF-Vektorisierung + Klassifikator.
    class_weight='balanced' adressiert das Klassenungleichgewicht (~24% Clickbait).
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),      # uni- und bigrams, fängt Phrasen wie "you won't" ein
        max_features=10000,
        min_df=2,
    )
    if model == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    elif model == "svm":
        clf = LinearSVC(class_weight="balanced")
    else:
        raise ValueError(f"Unbekanntes Modell: {model}")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_and_eval(train_df, test_df, model: str = "logreg"):
    """Trainiert die Pipeline und gibt Metriken auf dem Testset zurück."""
    pipe = build_pipeline(model)
    pipe.fit(train_df["headline"], train_df["label"])
    preds = pipe.predict(test_df["headline"])

    report = classification_report(
        test_df["label"], preds,
        target_names=["kein Clickbait", "Clickbait"], digits=3,
    )
    cm = confusion_matrix(test_df["label"], preds)
    f1 = f1_score(test_df["label"], preds)
    return pipe, {"report": report, "confusion_matrix": cm, "f1": f1}


if __name__ == "__main__":
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_csv, test_csv = data_dir / "train.csv", data_dir / "test.csv"

    if not train_csv.exists():
        print("[!] Bitte zuerst data_prep.py ausführen.")
    else:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        for model in ["logreg", "svm"]:
            print(f"\n===== Baseline: {model} =====")
            _, res = train_and_eval(train_df, test_df, model)
            print(res["report"])
            print("Konfusionsmatrix:\n", res["confusion_matrix"])
