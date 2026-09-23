"""
data_prep.py
------------
Lädt den Webis-Clickbait-Corpus 2017, verbindet instances.jsonl mit truth.jsonl,
reduziert die 4-Punkt-Annotation auf ein binäres Label und erstellt
stratifizierte Train/Val/Test-Splits.

Webis-Format (aus instances.jsonl):
    {"id": "...", "postText": ["Der eigentliche Tweet-Text"], "targetTitle": "...", ...}
Webis-Format (aus truth.jsonl):
    {"id": "...", "truthMean": 0.66, "truthClass": "clickbait", ...}

Ein Post gilt als Clickbait, wenn truthClass == "clickbait" (entspricht truthMean > 0.5).
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_webis(instances_path: str, truth_path: str) -> pd.DataFrame:
    """Verbindet instances.jsonl und truth.jsonl über die id zu einem DataFrame."""
    # instances laden
    instances = {}
    with open(instances_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # postText ist eine Liste; wir nehmen den zusammengefügten Text als Headline
            post_text = " ".join(obj.get("postText", [])).strip()
            instances[obj["id"]] = {
                "id": obj["id"],
                "headline": post_text,
                "target_title": obj.get("targetTitle", ""),
            }

    # truth laden und labeln
    rows = []
    with open(truth_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            _id = obj["id"]
            if _id not in instances:
                continue
            row = instances[_id]
            row["truth_mean"] = obj.get("truthMean", None)
            # binäres Label: 1 = Clickbait, 0 = kein Clickbait
            row["label"] = 1 if obj.get("truthClass") == "clickbait" else 0
            rows.append(row)

    df = pd.DataFrame(rows)
    # leere Headlines entfernen (Datenqualität)
    df = df[df["headline"].str.len() > 0].reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame, seed: int = 42):
    """
    Stratifizierter 70/15/15 Split. stratify sorgt dafür, dass das
    Klassenverhältnis (~24% Clickbait) in allen drei Teilen erhalten bleibt.
    """
    train, temp = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp["label"]
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def class_distribution(df: pd.DataFrame) -> dict:
    """Gibt die Klassenverteilung als Anteil zurück - für die EDA und den Report."""
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    return {
        "n": total,
        "clickbait": counts.get(1, 0),
        "no_clickbait": counts.get(0, 0),
        "clickbait_ratio": round(counts.get(1, 0) / total, 4) if total else 0.0,
    }


if __name__ == "__main__":
    # Beispiel-Aufruf - Pfade an euren Download anpassen
    data_dir = Path(__file__).resolve().parent.parent / "data"
    instances = data_dir / "instances.jsonl"
    truth = data_dir / "truth.jsonl"

    if not instances.exists():
        print(f"[!] {instances} nicht gefunden. Lade den Webis-Corpus von "
              f"https://zenodo.org/record/5530410 herunter und lege die "
              f"entpackten .jsonl-Dateien in data/ ab.")
    else:
        df = load_webis(str(instances), str(truth))
        print("Gesamt:", class_distribution(df))
        train, val, test = make_splits(df)
        for name, part in [("train", train), ("val", val), ("test", test)]:
            print(name, class_distribution(part))
            part.to_csv(data_dir / f"{name}.csv", index=False)
        print("Splits als CSV in data/ gespeichert.")
