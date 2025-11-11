#!/usr/bin/env python3
"""Summarize pipeline outputs and validate predictions vs labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class SequenceStats:
    total_sequences: int
    labeled_normals: Optional[int]
    labeled_anomalies: Optional[int]


@dataclass
class PredictionStats:
    total_predictions: int
    anomalies_flagged: int
    top_anomalies: list[Dict[str, Any]]


@dataclass
class EvaluationStats:
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize pipeline outputs and compute simple evaluation metrics."
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        default=Path("preprocessed/HDFS_sequences.npz"),
        help="NPZ file produced by build_sequences.py (contains x_data/y_data).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/isolation_forest_predictions.csv"),
        help="CSV of predictions produced by isolation_forest_inference.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pipeline_run_summary.json"),
        help="Destination JSON file for the summary (directories created automatically).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Include the top-k anomaly scores in the summary (default: 5).",
    )
    return parser.parse_args()


def load_sequences(npz_path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Sequence artifact not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        x_data = data["x_data"]
        y_data = data.get("y_data")
    return x_data, y_data


def compute_sequence_stats(x_data: np.ndarray, y_data: Optional[np.ndarray]) -> SequenceStats:
    total = int(len(x_data))
    if y_data is None:
        return SequenceStats(total_sequences=total, labeled_normals=None, labeled_anomalies=None)
    y_series = pd.Series(y_data)
    normals = int((y_series == 0).sum())
    anomalies = int((y_series == 1).sum())
    return SequenceStats(total_sequences=total, labeled_normals=normals, labeled_anomalies=anomalies)


def load_predictions(predictions_path: Path) -> tuple[pd.DataFrame, str]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")
    df = pd.read_csv(predictions_path)
    required_columns = {"prediction", "anomaly_score"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV is missing columns: {sorted(missing)}")
    if "score_direction" in df.columns:
        direction = df["score_direction"].iloc[0]
    else:
        direction = "lower"
    return df, str(direction)


def compute_prediction_stats(df: pd.DataFrame, top_k: int, score_direction: str) -> PredictionStats:
    anomalies_flagged = int((df["prediction"] == 1).sum())
    ascending = score_direction != "higher"
    top_rows = (
        df[df["prediction"] == 1]
        .sort_values("anomaly_score", ascending=ascending)
        .head(top_k)
    )
    top_payload = top_rows.to_dict(orient="records")
    return PredictionStats(
        total_predictions=int(len(df)),
        anomalies_flagged=anomalies_flagged,
        top_anomalies=top_payload,
    )


def compute_evaluation_stats(predictions: pd.DataFrame, labels: Optional[np.ndarray]) -> Optional[EvaluationStats]:
    if labels is None:
        return None
    if len(predictions) != len(labels):
        raise ValueError(
            "Prediction count does not match the number of labeled sequences. "
            f"predictions={len(predictions)}, labels={len(labels)}"
        )
    y_pred = predictions["prediction"].to_numpy()
    y_true = labels.astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def safe_div(num: int, denom: int) -> Optional[float]:
        return round(num / denom, 4) if denom > 0 else None

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    else:
        f1 = None

    return EvaluationStats(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def main() -> None:
    args = parse_args()

    x_data, y_data = load_sequences(args.sequences)
    seq_stats = compute_sequence_stats(x_data, y_data)

    predictions_df, score_direction = load_predictions(args.predictions)
    pred_stats = compute_prediction_stats(
        predictions_df, top_k=args.top_k, score_direction=score_direction
    )
    eval_stats = compute_evaluation_stats(predictions_df, y_data)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "sequences": str(args.sequences),
            "predictions": str(args.predictions),
        },
        "sequence_stats": asdict(seq_stats),
        "prediction_stats": {
            **asdict(pred_stats),
            "score_direction": score_direction,
        },
        "evaluation": asdict(eval_stats) if eval_stats is not None else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
