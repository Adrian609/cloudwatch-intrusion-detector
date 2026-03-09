"""Train an intrusion detector for tabular CloudWatch-style log data.

This script covers:
- Data loading and cleaning
- Missing-value handling
- Categorical encoding + numeric normalization
- Train/validation/test split
- Neural network training (PyTorch)
- Baseline model training (Logistic Regression)
- Metrics comparison and failure-case export

Example:
    python src/train.py --data data/your_dataset.csv --target label
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train intrusion detection models.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to a CSV dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name. If omitted, the script tries to infer it.",
    )
    parser.add_argument(
        "--positive-labels",
        type=str,
        default=None,
        help=(
            "Comma-separated class labels to treat as malicious (class 1) "
            "when target is categorical."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for artifacts and metrics.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying malicious traffic.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Max NN epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="NN batch size.",
    )
    parser.add_argument(
        "--max-cat-cardinality",
        type=int,
        default=100,
        help=(
            "Drop categorical columns with unique values above this number to avoid "
            "explosive one-hot encoding."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience on validation loss.",
    )
    return parser.parse_args()


def load_csv_robust(path: Path) -> pd.DataFrame:
    """Load CSV and recover from inconsistent row widths when needed."""
    try:
        return pd.read_csv(path, skipinitialspace=True)
    except pd.errors.ParserError:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            raw_header = next(reader)
            header = [h.strip() for h in raw_header if h.strip()]
            rows: list[list[str]] = []
            for row in reader:
                if len(row) < len(header):
                    continue
                trimmed = [cell.strip() for cell in row[: len(header)]]
                rows.append(trimmed)
        if not header:
            raise ValueError("Could not parse CSV header.")
        df = pd.DataFrame(rows, columns=header)
        return df.replace({"": np.nan})


def infer_target_column(df: pd.DataFrame, user_target: str | None) -> str:
    if user_target:
        if user_target not in df.columns:
            raise ValueError(f"Target column '{user_target}' not found in dataset.")
        return user_target

    candidates = [
        "label",
        "target",
        "class",
        "is_attack",
        "attack",
        "malicious",
        "intrusion",
        "threat",
        "detection_type",
        "detection_types",
        "attack_type",
        "attack_label",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    raise ValueError(
        "Could not infer target column. Pass --target <column_name> explicitly."
    )


def build_binary_target(
    raw_target: pd.Series,
    positive_labels: set[str] | None,
) -> tuple[pd.Series, str]:
    target = raw_target.copy()
    if target.dtype.kind in "bifc":
        unique_vals = sorted(pd.Series(target.dropna().unique()).tolist())
        unique_set = set(unique_vals)

        if unique_set <= {0, 1}:
            return target.astype(int), "Numeric target already binary (0/1)."

        if len(unique_vals) == 2:
            small, large = unique_vals[0], unique_vals[1]
            mapped = target.map({small: 0, large: 1}).astype(int)
            return mapped, f"Mapped numeric classes {small}->0 and {large}->1."

        mapped = (target.fillna(0) > 0).astype(int)
        return mapped, "Mapped numeric target using rule: value > 0 => malicious (1)."

    normalized = target.astype(str).str.strip().str.lower()
    label_values = sorted(normalized.unique().tolist())

    if positive_labels:
        pos = {x.strip().lower() for x in positive_labels}
        mapped = normalized.isin(pos).astype(int)
        return mapped, f"Used user-supplied positive labels: {sorted(pos)}."

    benign_tokens = {
        "benign",
        "normal",
        "legitimate",
        "legit",
        "safe",
        "allow",
        "allowed",
        "false",
        "negative",
        "no",
        "clean",
        "0",
    }

    if len(label_values) == 2:
        a, b = label_values
        a_benign = a in benign_tokens
        b_benign = b in benign_tokens
        if a_benign and not b_benign:
            mapped = normalized.map({a: 0, b: 1}).astype(int)
            return mapped, f"Mapped categorical classes {a}->0 and {b}->1."
        if b_benign and not a_benign:
            mapped = normalized.map({b: 0, a: 1}).astype(int)
            return mapped, f"Mapped categorical classes {b}->0 and {a}->1."

        mapped = normalized.map({a: 0, b: 1}).astype(int)
        return (
            mapped,
            "Mapped two categorical classes in sorted order (verify this mapping).",
        )

    mapped = (~normalized.isin(benign_tokens)).astype(int)
    return mapped, "Mapped multiclass labels to binary: benign-token labels vs others."


def safe_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    X_train: pd.DataFrame, max_cat_cardinality: int
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    dropped_high_card = []
    kept_categorical = []
    for col in categorical_cols:
        nunique = X_train[col].nunique(dropna=True)
        if nunique > max_cat_cardinality:
            dropped_high_card.append(col)
        else:
            kept_categorical.append(col)

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", safe_ohe()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, kept_categorical),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, kept_categorical, dropped_high_card


def densify_if_needed(X):
    return X.toarray() if hasattr(X, "toarray") else X


class FeedforwardNN(nn.Module):
    """Fully connected network with <=3 hidden layers and sigmoid output."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_nn(input_dim: int) -> FeedforwardNN:
    return FeedforwardNN(input_dim=input_dim)


def train_torch_nn(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    epochs: int,
    batch_size: int,
    patience: int = 5,
    learning_rate: float = 1e-3,
) -> dict[str, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bce = nn.BCELoss(reduction="none")
    w0 = torch.tensor(float(class_weights[0]), dtype=torch.float32, device=device)
    w1 = torch.tensor(float(class_weights[1]), dtype=torch.float32, device=device)

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            probs = model(xb)

            sample_weights = torch.where(yb >= 0.5, w1, w0)
            loss = (bce(probs, yb) * sample_weights).mean()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = (probs >= 0.5).float()
            running_correct += int((preds == yb).sum().item())
            running_total += int(xb.size(0))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_t)
            val_loss = float(nn.BCELoss()(val_probs, y_val_t).item())
            val_acc = float(((val_probs >= 0.5).float() == y_val_t).float().mean().item())

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"- loss: {train_loss:.4f} - acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict_torch_probs(model: nn.Module, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    probs_list: list[np.ndarray] = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            probs = model(xb).squeeze(1).detach().cpu().numpy()
            probs_list.append(probs)

    return np.concatenate(probs_list) if probs_list else np.array([], dtype=float)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def plot_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get("loss", []), label="train_loss")
    axes[0].plot(history.get("val_loss", []), label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="train_acc")
    axes[1].plot(history.get("val_accuracy", []), label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def categorize_failure(prob: float, missing_ratio: float, y_true: int) -> str:
    if missing_ratio >= 0.3:
        return "noise"
    if abs(prob - 0.5) <= 0.1:
        return "ambiguity"
    if (y_true == 0 and prob >= 0.9) or (y_true == 1 and prob <= 0.1):
        return "possible_labeling_error"
    return "class_overlap"


def export_failure_cases(
    X_test_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_csv: Path,
    max_rows: int = 5,
) -> int:
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        pd.DataFrame(columns=["note"]).to_csv(output_csv, index=False)
        return 0

    rows = X_test_raw.iloc[mis_idx].copy().reset_index(drop=True)
    rows["true_label"] = y_true[mis_idx]
    rows["pred_label"] = y_pred[mis_idx]
    rows["pred_prob_malicious"] = y_prob[mis_idx]
    rows["missing_ratio"] = rows.isna().mean(axis=1)
    rows["failure_type"] = rows.apply(
        lambda r: categorize_failure(
            prob=float(r["pred_prob_malicious"]),
            missing_ratio=float(r["missing_ratio"]),
            y_true=int(r["true_label"]),
        ),
        axis=1,
    )

    export_df = rows.head(max_rows).copy()
    export_df.to_csv(output_csv, index=False)
    return len(mis_idx)


def ensure_binary(y: Iterable[int]) -> None:
    unique_vals = set(pd.Series(y).dropna().unique().tolist())
    if not unique_vals <= {0, 1}:
        raise ValueError(f"Target is not binary after mapping. Unique values: {unique_vals}")


def main() -> None:
    args = parse_args()
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_csv_robust(args.data)
    original_rows = len(df)
    df.columns = [str(c).strip() for c in df.columns]
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    df = df.drop_duplicates().reset_index(drop=True)

    target_col = infer_target_column(df, args.target)
    positive_labels = None
    if args.positive_labels:
        positive_labels = {x.strip().lower() for x in args.positive_labels.split(",") if x.strip()}

    y, target_mapping_note = build_binary_target(df[target_col], positive_labels)
    ensure_binary(y)
    class_counts = y.value_counts()
    if class_counts.size < 2:
        raise ValueError(
            f"Target column '{target_col}' has only one class after mapping. "
            "Use a target/dataset that includes both benign (0) and malicious (1) examples."
        )
    X = df.drop(columns=[target_col]).copy()

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.1765,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    preprocessor, numeric_cols, categorical_cols, dropped_high_card = build_preprocessor(
        X_train, args.max_cat_cardinality
    )

    X_train_t = densify_if_needed(preprocessor.fit_transform(X_train))
    X_val_t = densify_if_needed(preprocessor.transform(X_val))
    X_test_t = densify_if_needed(preprocessor.transform(X_test))

    X_train_t = np.asarray(X_train_t, dtype=np.float32)
    X_val_t = np.asarray(X_val_t, dtype=np.float32)
    X_test_t = np.asarray(X_test_t, dtype=np.float32)

    y_train_arr = np.asarray(y_train, dtype=np.float32)
    y_val_arr = np.asarray(y_val, dtype=np.float32)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    class_values = np.array([0, 1], dtype=np.int64)
    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=class_values, y=np.asarray(y_train, dtype=np.int64)
    )
    class_weights = {0: float(class_weights_arr[0]), 1: float(class_weights_arr[1])}

    nn_model = build_nn(X_train_t.shape[1])
    history = train_torch_nn(
        model=nn_model,
        X_train=X_train_t,
        y_train=y_train_arr,
        X_val=X_val_t,
        y_val=y_val_arr,
        class_weights=class_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    nn_probs = predict_torch_probs(nn_model, X_test_t, batch_size=max(args.batch_size, 1024))
    nn_preds = (nn_probs >= args.threshold).astype(int)
    nn_metrics = compute_metrics(y_test_arr, nn_preds)

    baseline = LogisticRegression(max_iter=3000, class_weight="balanced")
    baseline.fit(X_train_t, np.asarray(y_train, dtype=np.int64))
    baseline_probs = baseline.predict_proba(X_test_t)[:, 1]
    baseline_preds = (baseline_probs >= args.threshold).astype(int)
    baseline_metrics = compute_metrics(y_test_arr, baseline_preds)

    torch.save(nn_model.state_dict(), args.output_dir / "nn_model.pt")
    joblib.dump(baseline, args.output_dir / "baseline_logreg.joblib")
    joblib.dump(preprocessor, args.output_dir / "preprocessor.joblib")

    plot_training_curves(history, args.output_dir / "training_curves.png")
    misclassified_total = export_failure_cases(
        X_test_raw=X_test,
        y_true=y_test_arr,
        y_pred=nn_preds,
        y_prob=nn_probs,
        output_csv=args.output_dir / "failure_cases.csv",
        max_rows=5,
    )

    comparison_df = pd.DataFrame(
        [
            {"model": "NeuralNetwork", **nn_metrics},
            {"model": "LogisticRegression", **baseline_metrics},
        ]
    )
    comparison_df.to_csv(args.output_dir / "model_comparison.csv", index=False)

    metadata = {
        "rows_original": int(original_rows),
        "rows_after_dedup": int(len(df)),
        "target_column": target_col,
        "target_mapping_note": target_mapping_note,
        "constant_columns_dropped": constant_cols,
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "high_cardinality_categorical_dropped": dropped_high_card,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "misclassified_total_nn": int(misclassified_total),
    }

    with open(args.output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(args.output_dir / "metrics_nn.json", "w", encoding="utf-8") as f:
        json.dump(nn_metrics, f, indent=2)
    with open(args.output_dir / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2)

    print("Training complete.")
    print(f"Target column: {target_col}")
    print(f"Target mapping: {target_mapping_note}")
    print(f"Neural network metrics: {nn_metrics}")
    print(f"Baseline metrics: {baseline_metrics}")
    print(f"Artifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
