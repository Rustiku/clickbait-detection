"""
transformer_clf.py
------------------
Fine-Tuning von BERT, RoBERTa und ModernBERT für die binäre Clickbait-Klassifikation.

Entspricht exakt der Trainingslogik in notebooks/Clickbait_Transformer_Finetuning.ipynb,
mit der die berichteten Ergebnisse erzeugt wurden (Google Colab, NVIDIA Tesla T4).

Modelle:
    BERT        -> bert-base-uncased
    RoBERTa     -> roberta-base
    ModernBERT  -> answerdotai/ModernBERT-base

Aufruf:
    python src/transformer_clf.py                 # alle drei Modelle, Seed 42
    python src/transformer_clf.py --seeds 42 1337 2024

Benötigt eine GPU. Modell-Checkpoints werden unter checkpoints/ abgelegt
(nicht versioniert), Ergebnisse unter results/.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODELS = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "modernbert": "answerdotai/ModernBERT-base",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": accuracy_score(labels, preds), "precision": p, "recall": r, "f1": f1}


class WeightedTrainer(Trainer):
    """Trainer mit klassengewichtetem Loss gegen das Klassenungleichgewicht (~24 % Clickbait)."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.CrossEntropyLoss(weight=w)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    n = len(df)
    n_pos = int(df["label"].sum())
    n_neg = n - n_pos
    return torch.tensor([n / (2 * n_neg), n / (2 * n_pos)], dtype=torch.float)


def run_experiment(model_name, train_df, val_df, test_df,
                   epochs=3, batch_size=16, lr=2e-5, max_length=64,
                   use_class_weights=True, seed=42):
    seed_everything(seed)
    short = model_name.split("/")[-1]
    print(f"\n{'=' * 60}\n  {model_name}  (seed {seed})\n{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["headline"], truncation=True, max_length=max_length)

    def to_ds(df):
        ds = Dataset.from_pandas(df[["headline", "label"]], preserve_index=False)
        return ds.map(tokenize, batched=True)

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    args = TrainingArguments(
        output_dir=f"./checkpoints/{short}-seed{seed}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        class_weights=get_class_weights(train_df) if use_class_weights else None,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    n_params = sum(p.numel() for p in model.parameters())

    print(classification_report(test_df["label"], preds,
                                target_names=["kein Clickbait", "Clickbait"], digits=3))
    print("Konfusionsmatrix:\n", confusion_matrix(test_df["label"], preds))
    print(f"Trainingszeit: {train_time:.1f}s | Parameter: {n_params / 1e6:.1f} Mio")

    result = {
        "model": model_name,
        "seed": seed,
        "test_f1": test_metrics["test_f1"],
        "test_precision": test_metrics["test_precision"],
        "test_recall": test_metrics["test_recall"],
        "test_accuracy": test_metrics["test_accuracy"],
        "train_time_sec": round(train_time, 1),
        "n_params_mio": round(n_params / 1e6, 1),
    }
    return result, preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_df = pd.read_csv(root / "data" / "train.csv")
    val_df = pd.read_csv(root / "data" / "val.csv")
    test_df = pd.read_csv(root / "data" / "test.csv")
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)

    all_results = []
    for key in args.models:
        for seed in args.seeds:
            res, preds = run_experiment(MODELS[key], train_df, val_df, test_df, seed=seed)
            res["key"] = key
            all_results.append(res)
            pd.DataFrame({"id": test_df["id"], "label": test_df["label"], "pred": preds}).to_csv(
                out_dir / f"predictions_{key}_seed{seed}.csv", index=False)

    with open(out_dir / "transformer_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nGespeichert: results/transformer_results.json und results/predictions_*.csv")
