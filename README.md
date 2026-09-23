# Clickbait Headline Detection using BERT and RoBERTa

Vergleich klassischer und Transformer-basierter Verfahren zur binären
Clickbait-Erkennung auf dem Webis-Clickbait-Corpus 2017. Projekt im Rahmen der
Übung Informationslinguistik 2 (Lehrstuhl für Informationswissenschaft,
Universität Regensburg).

**Autoren:** Rustem Kuduschev (2184638), Tim Frummet (2388999)

## Überblick

Unter identischen Bedingungen (gleiche Splits, Klassengewichtung und
Hyperparameter) werden verglichen:

- **Baseline:** TF-IDF + Logistic Regression / lineare SVM
- **Transformer:** BERT, RoBERTa, ModernBERT (Fine-Tuning)

| Modell | F1 Clickbait (Testset, Seed 42) | F1 mit Streuung |
|---|---|---|
| TF-IDF + LogReg | 0,538 | 5-fold CV: 0,536 ± 0,017 |
| TF-IDF + SVM | 0,543 | 5-fold CV: 0,525 ± 0,020 |
| BERT | 0,692 | 0,688 ± 0,004 |
| RoBERTa | 0,690 | 0,701 ± 0,013 |
| ModernBERT | 0,695 | 0,702 ± 0,007 |

## Datensatz

Webis-Clickbait-Corpus 2017 (Potthast et al., 2018), Trainingsteil
`clickbait17-train-170630.zip`: https://zenodo.org/record/5530410

Der Datensatz ist nicht im Repository enthalten. Nach dem Download die Dateien
`instances.jsonl` und `truth.jsonl` nach `data/` kopieren.

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduktion

```bash
python src/data_prep.py            # Label-Ableitung, 70/15/15-Split (seed 42) -> data/*.csv
python src/eda.py                  # sprachliche Merkmale nach Klasse
python src/baseline.py             # TF-IDF-Baselines, Testset
python src/baseline_cv.py          # 5-fold CV der Baselines
python src/transformer_clf.py --seeds 42 1337 2024   # Fine-Tuning (GPU erforderlich)
python src/label_noise.py          # Fehlerquote nach Annotationsübereinstimmung
```

Das Fine-Tuning wurde auf Google Colab (NVIDIA Tesla T4) mit dem Notebook
`notebooks/Clickbait_Transformer_Finetuning.ipynb` durchgeführt; es enthält alle
Ausgaben der berichteten Läufe. `src/transformer_clf.py` enthält dieselbe
Trainingslogik als eigenständiges Skript.

Alle Zufallsprozesse sind geseedet. Wiederholte Läufe auf derselben Hardware
ergaben identische Ergebnisse.

## Struktur

```
clickbait-detection/
├── data/                  # Webis-Rohdaten und Splits (nicht versioniert)
├── notebooks/
│   └── Clickbait_Transformer_Finetuning.ipynb
├── results/               # alle Ergebnisdateien der berichteten Läufe
├── src/
│   ├── data_prep.py       # Laden, Label-Ableitung, stratifizierter Split
│   ├── eda.py             # explorative Datenanalyse
│   ├── baseline.py        # TF-IDF + LogReg / SVM
│   ├── baseline_cv.py     # 5-fold Cross-Validation der Baselines
│   ├── transformer_clf.py # Fine-Tuning BERT / RoBERTa / ModernBERT
│   ├── error_analysis.py  # Fehlerkategorisierung
│   └── label_noise.py     # Fehlerquote nach truth_mean-Bereich
├── requirements.txt
└── README.md
```

## Hinweise

- Das Fine-Tuning ohne GPU ist unpraktikabel langsam.
- ModernBERT benötigt `transformers>=4.48`. Die Experimente liefen mit
  `transformers` 5.x.
