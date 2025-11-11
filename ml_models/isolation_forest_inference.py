"""Run Isolation Forest inference on precomputed HDFS sequences.

Example:
    python -m ml_models.isolation_forest_inference \
        --data-path preprocessed/HDFS_sequences.npz \
        --vocab-path ml_models/isolation_forest_event_vocab.json \
        --model-path ml_models/isolation_forest_model.joblib \
        --block-index-path preprocessed/HDFS_block_index.csv \
        --output-path outputs/isolation_forest_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

DEFAULT_DATA_PATH = Path("preprocessed/HDFS_sequences.npz")
DEFAULT_MODEL_PATH = Path("ml_models/isolation_forest_model.joblib")
DEFAULT_VOCAB_PATH = Path("ml_models/isolation_forest_event_vocab.json")
DEFAULT_BLOCK_INDEX_PATH = Path("preprocessed/HDFS_block_index.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/isolation_forest_predictions.csv")
DEFAULT_THRESHOLD_PATH = Path("ml_models/isolation_forest_threshold.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained Isolation Forest model and generate anomaly predictions "
            "for log block sequences represented in an NPZ file."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="NPZ file with x_data sequences.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained Isolation Forest .joblib artifact.",
    )

    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=DEFAULT_VOCAB_PATH,
        help="JSON file containing the event vocabulary saved during training.",
    )
    parser.add_argument(
        "--block-index-path",
        type=Path,
        default=DEFAULT_BLOCK_INDEX_PATH,
        help=(
            "Optional CSV produced by scripts/build_sequences.py containing BlockId per sequence. "
            "If missing, sequence indices will be used instead."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV for predictions and anomaly scores.",
    )
    parser.add_argument(
        "--threshold-path",
        type=Path,
        default=DEFAULT_THRESHOLD_PATH,
        help=(
            "Optional JSON file describing the calibrated anomaly score threshold. "
            "If not provided or missing, the raw Isolation Forest predictions are used."
        ),
    )
    return parser.parse_args()


def load_sequences(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Sequence file '{npz_path}' not found.")
    with np.load(npz_path, allow_pickle=True) as data:
        return data["x_data"]


def load_vocab(vocab_path: Path) -> dict[str, int]:
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Event vocabulary file '{vocab_path}' not found. Run isolation_forest_training.py first."
        )
    with vocab_path.open("r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    # Keys are event IDs; ensure indices are ints and sort by index to validate shape.
    mapping = {str(event): int(idx) for event, idx in raw_mapping.items()}
    ordered = sorted(mapping.items(), key=lambda item: item[1])
    for expected, (event, idx) in enumerate(ordered):
        if idx != expected:
            raise ValueError(
                "Vocabulary indices are not contiguous starting at 0. "
                "Did the mapping file come from the current training run?"
            )
    return mapping


def load_threshold(threshold_path: Path | None) -> dict | None:
    if threshold_path is None:
        return None
    if not threshold_path.exists():
        return None
    with threshold_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    threshold = payload.get("score_threshold")
    if threshold is None:
        raise ValueError(
            "Threshold JSON is missing the 'score_threshold' field. "
            "Re-run training to regenerate it."
        )
    payload["score_threshold"] = float(threshold)
    return payload


def build_feature_matrix(sequences: Iterable[Iterable[str]], mapping: dict[str, int]):
    matrix = lil_matrix((len(sequences), len(mapping)), dtype=int)
    unknown_events: set[str] = set()
    for row_idx, seq in enumerate(sequences):
        for event in seq:
            idx = mapping.get(event)
            if idx is not None:
                matrix[row_idx, idx] += 1
            else:
                unknown_events.add(str(event))
    return matrix.tocsr(), unknown_events


def load_block_index(block_index_path: Path, expected_rows: int) -> pd.Series:
    if not block_index_path.exists():
        return pd.Series([f"sequence_{i}" for i in range(expected_rows)], name="SequenceId")
    df = pd.read_csv(block_index_path)
    if "BlockId" not in df.columns:
        raise ValueError("Block index CSV must contain a 'BlockId' column.")
    if len(df) != expected_rows:
        raise ValueError(
            f"Block index row count ({len(df)}) does not match number of sequences ({expected_rows})."
        )
    return df["BlockId"].rename("BlockId")


def main() -> None:
    args = parse_args()

    sequences = load_sequences(args.data_path)
    mapping = load_vocab(args.vocab_path)
    feature_matrix, unknown_events = build_feature_matrix(sequences, mapping)

    model = joblib.load(args.model_path)
    if feature_matrix.shape[1] != getattr(model, "n_features_in_", feature_matrix.shape[1]):
        raise ValueError(
            "Feature dimension mismatch between sequences and trained model. "
            "Ensure you're using the vocabulary from the same training run."
        )

    threshold_config = load_threshold(args.threshold_path)

    if threshold_config is not None:
        algorithm = threshold_config.get("algorithm") or "isolation-forest"
        score_direction = threshold_config.get("score_direction", "lower")
        threshold_value = threshold_config["score_threshold"]
    else:
        algorithm = getattr(model, "__class__", type("", (), {}))
        algorithm = (
            "random-forest"
            if "Classifier" in algorithm.__name__ and hasattr(model, "predict_proba")
            else "isolation-forest"
        )
        score_direction = "lower"
        threshold_value = 0.0 if algorithm == "isolation-forest" else 0.5

    if algorithm == "random-forest":
        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "Loaded model does not support predict_proba but algorithm was random-forest."
            )
        anomaly_scores = model.predict_proba(feature_matrix)[:, 1]
        prediction_labels = np.where(anomaly_scores >= threshold_value, 1, 0)
    else:
        if not hasattr(model, "decision_function"):
            raise ValueError(
                "Loaded model is missing decision_function; ensure it is an Isolation Forest."
            )
        anomaly_scores = model.decision_function(feature_matrix)
        if score_direction == "higher":
            prediction_labels = np.where(anomaly_scores >= threshold_value, 1, 0)
        else:
            prediction_labels = np.where(anomaly_scores < threshold_value, 1, 0)

    block_ids = load_block_index(args.block_index_path, len(sequences))
    output_df = pd.DataFrame(
        {
            block_ids.name or "BlockId": block_ids,
            "prediction": prediction_labels,
            "anomaly_score": anomaly_scores,
        }
    )

    output_df["decision_threshold"] = threshold_value
    output_df["score_direction"] = score_direction
    output_df["algorithm"] = algorithm

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_path, index=False)

    print(f"Predictions saved to '{args.output_path}'")
    print(
        f"Anomalies flagged: {int(prediction_labels.sum())} / {len(prediction_labels)}"
    )
    if unknown_events:
        print(
            "Warning: encountered events not in training vocabulary (ignored): "
            f"{len(unknown_events)} unique"
        )


if __name__ == "__main__":
    main()
