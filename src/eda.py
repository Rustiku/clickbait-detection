"""
eda.py
------
Explorative Datenanalyse der Clickbait-Headlines.

WICHTIG (aus dem Exposé-Feedback): Die hier untersuchten Merkmale sind NICHT als
eigene Neuentdeckung zu präsentieren. Was Clickbait sprachlich ausmacht, ist bereits
gut erforscht - im Report solltet ihr diese Analyse über etablierte Literatur rahmen,
insbesondere:
    Scott, K. (2021). You won't believe what's in this paper! Clickbait, relevance
    and the curiosity gap. Journal of Pragmatics.
Die EDA dient hier also der *Bestätigung bekannter Muster im konkreten Datensatz*,
nicht als Beitrag.
"""

import re
import pandas as pd

CLICKBAIT_TRIGGERS = [
    "you", "this", "what", "why", "how", "these", "reason", "reasons",
    "will", "won't", "believe", "shocking", "amazing", "secret",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet einfache sprachliche Merkmale pro Headline."""
    feats = pd.DataFrame()
    h = df["headline"].fillna("")

    feats["label"] = df["label"]
    feats["char_len"] = h.str.len()
    feats["word_len"] = h.str.split().apply(len)
    feats["has_number"] = h.str.contains(r"\d", regex=True).astype(int)
    feats["starts_with_number"] = h.str.match(r"^\s*\d").astype(int)
    feats["question_mark"] = h.str.contains(r"\?", regex=True).astype(int)
    feats["exclamation"] = h.str.contains(r"!", regex=True).astype(int)
    feats["trigger_count"] = h.str.lower().apply(
        lambda t: sum(1 for w in CLICKBAIT_TRIGGERS if w in t.split())
    )
    return feats


def summarize(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Vergleicht die Merkmale zwischen Clickbait (label=1) und Nicht-Clickbait (label=0).
    Diese Tabelle eignet sich direkt für den EDA-Abschnitt im Report.
    """
    grouped = feats.groupby("label").mean(numeric_only=True).T
    grouped.columns = ["kein_clickbait", "clickbait"]
    grouped["differenz"] = grouped["clickbait"] - grouped["kein_clickbait"]
    return grouped.round(3)


if __name__ == "__main__":
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_csv = data_dir / "train.csv"
    if not train_csv.exists():
        print("[!] Bitte zuerst data_prep.py ausführen, um train.csv zu erzeugen.")
    else:
        df = pd.read_csv(train_csv)
        feats = extract_features(df)
        print(summarize(feats))
