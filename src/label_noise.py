"""
label_noise.py
--------------
Prüft, ob Fehlklassifikationen gehäuft bei Posts auftreten, bei denen sich die
Annotator:innen uneinig waren.

truth_mean ist der Mittelwert der fünf Annotationen (0 = kein Clickbait,
1 = eindeutig Clickbait). Werte nahe 0,5 bedeuten geringe Übereinstimmung.

Entscheidend ist die FEHLERQUOTE pro Bereich, nicht der Anteil der Fehler:
Nur wenn Modelle im Grenzbereich deutlich häufiger irren als außerhalb,
stützt das die These, dass ein Teil der Fehler auf mehrdeutige Fälle zurückgeht.

Eingaben:
    data/test.csv                          alle Testbeispiele mit truth_mean
    results/error_analysis_<modell>.csv    Fehlklassifikationen je Modell (Spalte id)
"""

from pathlib import Path

import pandas as pd

BINS = [0.0, 0.2, 0.35, 0.65, 0.8, 1.0001]
LABELS = [
    "0,00–0,20 (klar kein CB)",
    "0,20–0,35",
    "0,35–0,65 (Grenzbereich)",
    "0,65–0,80",
    "0,80–1,00 (klar CB)",
]
GRENZ = LABELS[2]
MODEL_ORDER = ["bert", "roberta", "modernbert"]


def error_rate_by_agreement(test_df: pd.DataFrame, error_ids: set) -> pd.DataFrame:
    df = test_df.copy()
    df["is_error"] = df["id"].astype(str).isin(error_ids)
    df["bereich"] = pd.cut(df["truth_mean"], bins=BINS, labels=LABELS, right=False)
    table = df.groupby("bereich", observed=False).agg(
        n_test=("id", "size"), n_fehler=("is_error", "sum")
    )
    table["fehlerquote_%"] = (100 * table["n_fehler"] / table["n_test"]).round(1)
    return table


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    test_df = pd.read_csv(root / "data" / "test.csv")

    files = sorted((root / "results").glob("error_analysis_*.csv"))
    if not files:
        raise SystemExit("[!] Keine Dateien results/error_analysis_<modell>.csv gefunden.")

    models = sorted(
        (f.stem.replace("error_analysis_", "") for f in files),
        key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99,
    )

    rates, summary = {}, []
    n_test = None
    for m in models:
        err = pd.read_csv(root / "results" / f"error_analysis_{m}.csv")
        table = error_rate_by_agreement(test_df, set(err["id"].astype(str)))
        n_test = table["n_test"]
        rates[m] = table["fehlerquote_%"]

        n_err = int(table["n_fehler"].sum())
        n_grenz = int(table.loc[GRENZ, "n_fehler"])
        n_aussen = n_err - n_grenz
        summary.append({
            "modell": m,
            "fehler_gesamt": n_err,
            "fehlerquote_gesamt_%": round(100 * n_err / len(test_df), 1),
            "fehlerquote_grenzbereich_%": table.loc[GRENZ, "fehlerquote_%"],
            "fehlerquote_ausserhalb_%": round(
                100 * n_aussen / (len(test_df) - table.loc[GRENZ, "n_test"]), 1),
            "anteil_fehler_im_grenzbereich_%": round(100 * n_grenz / n_err, 1),
        })

    by_bin = pd.DataFrame({"n_test": n_test, **rates})
    summary_df = pd.DataFrame(summary).set_index("modell")

    print("Fehlerquote (%) nach Annotationsübereinstimmung:\n")
    print(by_bin.to_string())
    print("\nZusammenfassung:\n")
    print(summary_df.to_string())

    by_bin.to_csv(root / "results" / "label_noise_by_bin.csv")
    summary_df.to_csv(root / "results" / "label_noise_summary.csv")
    print("\nGespeichert: results/label_noise_by_bin.csv, results/label_noise_summary.csv")
