# %%
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split

"""Train an Isolation Forest log anomaly detector from pre-built HDFS sequences.

Example:
    python ml_models/isolation_forest_training.py \
        --data-path preprocessed/HDFS_sequences.npz \
        --model-output-path ml_models/isolation_forest_model.joblib \
        --vocab-output-path ml_models/isolation_forest_event_vocab.json

The input NPZ should be produced by `scripts/build_sequences.py` and contain
`x_data` and `y_data` arrays.
"""


DEFAULT_DATA_PATH = Path("preprocessed/HDFS_sequences.npz")
DEFAULT_MODEL_OUTPUT_PATH = Path("ml_models/isolation_forest_model.joblib")
DEFAULT_VOCAB_OUTPUT_PATH = Path("ml_models/isolation_forest_event_vocab.json")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training script."""
    parser = argparse.ArgumentParser(
        description=(
            "Train an Isolation Forest using precomputed HDFS event sequences. "
            "By default the script consumes sequences built by "
            "scripts/build_sequences.py and saves a fitted model artifact."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the NPZ file containing `x_data` and `y_data` arrays.",
    )
    parser.add_argument(
        "--model-output-path",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT_PATH,
        help="Destination for the trained Isolation Forest model (.joblib).",
    )
    parser.add_argument(
        "--vocab-output-path",
        type=Path,
        default=DEFAULT_VOCAB_OUTPUT_PATH,
        help="Where to save the JSON mapping of event IDs to column indices.",
    )
    parser.add_argument(
        "--skip-grid-search",
        action="store_true",
        help="Train with a sensible default configuration instead of running GridSearchCV.",
    )
    return parser.parse_args()


def load_sequences(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Sequence file '{npz_path}' not found. Run scripts/build_sequences.py first."
        )

    print(f"--> Loading sequence data from {npz_path}...")
    with np.load(npz_path, allow_pickle=True) as data:
        X_sequences = data["x_data"]
        y = data["y_data"]
    return X_sequences, y


def build_event_count_matrix(X_sequences: np.ndarray) -> tuple[lil_matrix, dict[str, int]]:
    print("--> Creating event count matrix from sequences...")
    all_events = [event for seq in X_sequences for event in seq]
    unique_events = sorted(list(set(all_events)))
    event_to_int = {event: i for i, event in enumerate(unique_events)}
    num_unique_events = len(unique_events)

    X_counts = lil_matrix((len(X_sequences), num_unique_events), dtype=int)
    for i, seq in enumerate(X_sequences):
        for event in seq:
            if event in event_to_int:
                X_counts[i, event_to_int[event]] += 1

    X_counts = X_counts.tocsr()

    print("\n--- Data Summary ---")
    print(f"Number of log blocks: {X_counts.shape[0]}")
    print(f"Number of unique events (features): {X_counts.shape[1]}")
    return X_counts, event_to_int


def contamination_ratio(labels: np.ndarray) -> float:
    ratio = float(labels.sum()) / float(len(labels)) if len(labels) else 0.0
    return max(ratio, 1e-6)  # Avoid zero contamination for Isolation Forest


def map_if_predictions(y_pred: np.ndarray) -> np.ndarray:
    return np.array([1 if pred == -1 else 0 for pred in y_pred])


def run_grid_search(X_train, y_train) -> GridSearchCV:
    print("\n--> Setting up GridSearchCV for hyperparameter tuning...")
    contamination = contamination_ratio(y_train)
    print(f"Calculated contamination for training set: {contamination:.4f}")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_samples": ["auto", 0.8, 1.0],
        "contamination": [contamination, 0.05, 0.1],
        "max_features": [0.5, 0.75, 1.0],
    }

    scorer = make_scorer(lambda y_true, y_pred: f1_score(y_true, map_if_predictions(y_pred)))
    iso_forest = IsolationForest(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=iso_forest,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=2,
    )

    print("--> Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best F1-score from cross-validation: {grid_search.best_score_:.4f}")
    return grid_search


def train_default_model(X_train, y_train) -> IsolationForest:
    print("\n--> Training Isolation Forest with default parameters...")
    contamination = contamination_ratio(y_train)
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    args = parse_args()
    start_time = time.time()

    X_sequences, y = load_sequences(args.data_path)
    X_counts, event_to_int = build_event_count_matrix(X_sequences)
    print(f"Number of anomalies: {int(y.sum())}")

    print("\n--> Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_counts, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    if args.skip_grid_search:
        model = train_default_model(X_train, y_train)
    else:
        grid_search = run_grid_search(X_train, y_train)
        model = grid_search.best_estimator_
        print("\nModel training with best parameters complete.")

    args.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output_path)
    print(f"Model saved to '{args.model_output_path}'")

    args.vocab_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.vocab_output_path.open("w", encoding="utf-8") as f:
        json.dump(event_to_int, f, indent=2, sort_keys=True)
    print(f"Event vocabulary saved to '{args.vocab_output_path}'")

    print("\n--> Evaluating model performance on the test set...")
    y_pred = model.predict(X_test)
    y_pred_mapped = map_if_predictions(y_pred)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred_mapped, target_names=["Normal", "Anomaly"]))

    end_time = time.time()
    print(f"\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
