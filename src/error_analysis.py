"""
error_analysis.py
-----------------
Qualitative Fehleranalyse: extrahiert falsch klassifizierte Headlines und
kategorisiert sie grob nach Clickbait-Typ. Liefert die Beispiele für den
Discussion-Teil des Reports (der Gutachter erwartet nicht nur Zahlen, sondern
konkrete Beispiele + Muster).
"""

import re
import pandas as pd


def categorize(headline: str) -> str:
    """Grobe heuristische Typisierung einer Headline nach Clickbait-Muster."""
    h = headline.lower()
    if re.match(r"^\s*\d+\s", headline) or re.search(r"\b\d+\s+(reasons|things|ways|photos)\b", h):
        return "listicle"
    if "?" in headline:
        return "rhetorical_question"
    if any(w in h.split() for w in ["you", "your", "you'll", "you're"]):
        return "direct_address"
    if any(w in h for w in ["this", "these", "what", "why"]):
        return "curiosity_gap"
    return "other"


def collect_errors(test_df: pd.DataFrame, predictions) -> pd.DataFrame:
    """
    Baut eine Tabelle aller Fehler (FP und FN) mit Typ-Kategorie.
    test_df muss die Spalten 'headline' und 'label' haben.
    """
    df = test_df.copy().reset_index(drop=True)
    df["pred"] = predictions
    errors = df[df["label"] != df["pred"]].copy()
    errors["error_type"] = errors.apply(
        lambda r: "false_positive" if r["pred"] == 1 else "false_negative", axis=1
    )
    errors["clickbait_pattern"] = errors["headline"].apply(categorize)
    return errors[["headline", "label", "pred", "error_type", "clickbait_pattern"]]


def error_summary(errors: pd.DataFrame) -> pd.DataFrame:
    """Zählt Fehler nach Muster und Fehlertyp - direkt für eine Tabelle im Report."""
    return (
        errors.groupby(["clickbait_pattern", "error_type"])
        .size()
        .unstack(fill_value=0)
    )


if __name__ == "__main__":
    from pathlib import Path
    from baseline import train_and_eval
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if not (data_dir / "train.csv").exists():
        print("[!] Bitte zuerst data_prep.py ausführen.")
    else:
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        pipe, _ = train_and_eval(train_df, test_df, "logreg")
        preds = pipe.predict(test_df["headline"])
        errors = collect_errors(test_df, preds)
        print(f"Anzahl Fehler: {len(errors)}")
        if len(errors):
            print(errors.to_string(index=False))
            print("\nZusammenfassung:")
            print(error_summary(errors))
