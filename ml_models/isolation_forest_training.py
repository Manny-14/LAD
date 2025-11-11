# %%
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, make_scorer, precision_recall_curve
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
DEFAULT_THRESHOLD_OUTPUT_PATH = Path("ml_models/isolation_forest_threshold.json")
ALGORITHM_CHOICES = ("isolation-forest", "random-forest")


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
        "--threshold-output-path",
        type=Path,
        default=DEFAULT_THRESHOLD_OUTPUT_PATH,
        help="Where to write the calibrated decision threshold JSON.",
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHM_CHOICES,
        default="isolation-forest",
        help=(
            "Training algorithm to use. 'random-forest' fits a supervised classifier (requires "
            "--allow-supervised), while 'isolation-forest' keeps the unsupervised detector."
        ),
    )
    parser.add_argument(
        "--allow-supervised",
        action="store_true",
        help="Acknowledge that you intend to train a supervised model (required for --algorithm random-forest).",
    )
    parser.add_argument(
        "--skip-grid-search",
        action="store_true",
        help="Train with a sensible default configuration instead of running GridSearchCV.",
    )
    parser.add_argument(
        "--calibration-beta",
        type=float,
        default=1.0,
        help=(
            "Beta value for F-beta scoring during threshold calibration. "
            "Values > 1 weight recall more heavily; values < 1 prefer precision."
        ),
    )
    parser.add_argument(
        "--min-calibration-precision",
        type=float,
        default=None,
        help=(
            "Optional lower bound on precision when selecting the calibration threshold. "
            "If no threshold satisfies the constraint, the best unconstrained value is used."
        ),
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


def split_for_calibration(
    X_train_full, y_train_full, calibration_fraction: float = 0.2
):
    """Split training data into model-training and calibration subsets.

    Falls back to using the full training data for calibration if class counts are too small
    for a stratified split (e.g., in unit tests with tiny synthetic datasets).
    """

    unique_labels, counts = np.unique(y_train_full, return_counts=True)
    if unique_labels.size < 2 or np.min(counts) < 2:
        print(
            "Insufficient class diversity for a separate calibration split; "
            "reusing full training data."
        )
        return X_train_full, X_train_full, y_train_full, y_train_full

    X_train, X_calibration, y_train, y_calibration = train_test_split(
        X_train_full,
        y_train_full,
        test_size=calibration_fraction,
        random_state=42,
        stratify=y_train_full,
    )
    return X_train, X_calibration, y_train, y_calibration


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    lower_scores_are_more_anomalous: bool,
    beta: float = 1.0,
    min_precision: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute the anomaly score threshold that maximizes F1 on calibration data."""
    if scores.size == 0 or labels.size == 0:
        return 0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    working_scores = -scores if lower_scores_are_more_anomalous else scores
    precision, recall, thresholds = precision_recall_curve(labels, working_scores)
    if thresholds.size == 0:
        # Fallback to default decision boundary at 0 if we only observed one class
        fallback_precision = float(precision[-1])
        fallback_recall = float(recall[-1])
        baseline_f1 = float(
            2 * fallback_precision * fallback_recall
            / max(fallback_precision + fallback_recall, 1e-12)
        )
        beta_sq = float(beta) ** 2
        baseline_f_beta = float(
            (1 + beta_sq)
            * fallback_precision
            * fallback_recall
            / max(beta_sq * fallback_precision + fallback_recall, 1e-12)
        )
        default_threshold = 0.0 if lower_scores_are_more_anomalous else 0.5
        return default_threshold, {
            "precision": fallback_precision,
            "recall": fallback_recall,
            "f1": baseline_f1,
            "f_beta": baseline_f_beta,
        }

    usable_precision = precision[:-1]
    usable_recall = recall[:-1]
    beta_sq = float(beta) ** 2
    f_beta_scores = (1 + beta_sq) * usable_precision * usable_recall / np.clip(
        beta_sq * usable_precision + usable_recall, 1e-12, None
    )

    if min_precision is not None:
        feasible = usable_precision >= min_precision
        if not np.any(feasible):
            print(
                "No calibration threshold satisfied the requested precision constraint; "
                "falling back to the best unconstrained threshold."
            )
        else:
            f_beta_scores = np.where(feasible, f_beta_scores, -np.inf)

    best_index = int(np.nanargmax(f_beta_scores))

    best_threshold = float(
        -thresholds[best_index]
        if lower_scores_are_more_anomalous
        else thresholds[best_index]
    )
    chosen_precision = float(usable_precision[best_index])
    chosen_recall = float(usable_recall[best_index])
    f1_score_value = float(
        2 * chosen_precision * chosen_recall
        / max(chosen_precision + chosen_recall, 1e-12)
    )
    metrics = {
        "precision": chosen_precision,
        "recall": chosen_recall,
        "f1": f1_score_value,
        "f_beta": float(f_beta_scores[best_index]),
    }
    return best_threshold, metrics


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


def train_isolation_forest_model(X_train, y_train) -> IsolationForest:
    print("\n--> Training Isolation Forest with default parameters...")
    contamination = contamination_ratio(y_train)
    model = IsolationForest(
        n_estimators=100,
        max_samples=1.0,
        contamination=contamination,
        max_features=0.75,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_model(X_train, y_train) -> RandomForestClassifier:
    print("\n--> Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    args = parse_args()
    if args.calibration_beta <= 0:
        raise ValueError("--calibration-beta must be positive.")
    if args.min_calibration_precision is not None and not (0.0 < args.min_calibration_precision < 1.0):
        raise ValueError("--min-calibration-precision must be between 0 and 1 (exclusive).")
    start_time = time.time()

    X_sequences, y = load_sequences(args.data_path)
    X_counts, event_to_int = build_event_count_matrix(X_sequences)
    print(f"Number of anomalies: {int(y.sum())}")

    print("\n--> Splitting data into training and testing sets...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_counts, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train_full.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    print("--> Reserving calibration split for threshold tuning...")
    X_train, X_calibration, y_train, y_calibration = split_for_calibration(
        X_train_full, y_train_full
    )
    print(f"Model training set size: {X_train.shape[0]}")
    print(f"Calibration set size: {X_calibration.shape[0]}")

    algorithm = args.algorithm
    if algorithm == "random-forest" and not args.allow_supervised:
        raise ValueError(
            "Random Forest is a supervised algorithm and typically yields near-perfect scores on HDFS. "
            "Re-run with --allow-supervised if you really want this baseline."
        )
    if algorithm == "isolation-forest":
        if args.skip_grid_search:
            model = train_isolation_forest_model(X_train, y_train)
        else:
            grid_search = run_grid_search(X_train, y_train)
            model = grid_search.best_estimator_
            print("\nModel training with best parameters complete.")
        calibration_scores = model.decision_function(X_calibration)
        test_scores = model.decision_function(X_test)
        lower_scores_are_more_anomalous = True
    elif algorithm == "random-forest":
        if not args.skip_grid_search:
            print(
                "[warning] Grid search is currently only supported for Isolation Forest; "
                "ignoring tuning request for Random Forest."
            )
        print(
            "[warning] Training a supervised Random Forest on labeled data; metrics will reflect a fully supervised baseline."
        )
        model = train_random_forest_model(X_train, y_train)
        calibration_scores = model.predict_proba(X_calibration)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        lower_scores_are_more_anomalous = False
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    args.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output_path)
    print(f"Model saved to '{args.model_output_path}'")

    args.vocab_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.vocab_output_path.open("w", encoding="utf-8") as f:
        json.dump(event_to_int, f, indent=2, sort_keys=True)
    print(f"Event vocabulary saved to '{args.vocab_output_path}'")

    print("\n--> Calibrating decision threshold on held-out data...")
    threshold, calibration_metrics = find_optimal_threshold(
        calibration_scores,
        y_calibration,
        lower_scores_are_more_anomalous=lower_scores_are_more_anomalous,
        beta=args.calibration_beta,
        min_precision=args.min_calibration_precision,
    )
    score_label = "F1" if abs(args.calibration_beta - 1.0) < 1e-6 else f"F{args.calibration_beta:.2f}"
    representative_score = (
        calibration_metrics["f1"]
        if abs(args.calibration_beta - 1.0) < 1e-6
        else calibration_metrics["f_beta"]
    )
    print(
        "Selected anomaly score threshold: "
        f"{threshold:.6f} ({score_label}={representative_score:.4f}, "
        f"precision={calibration_metrics['precision']:.4f}, "
        f"recall={calibration_metrics['recall']:.4f})"
    )

    args.threshold_output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.threshold_output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "algorithm": algorithm,
                "score_threshold": threshold,
                "score_direction": "lower" if lower_scores_are_more_anomalous else "higher",
                "calibration_metrics": calibration_metrics,
                "calibration_size": int(y_calibration.shape[0]),
                "calibration_beta": float(args.calibration_beta),
                "min_calibration_precision": (
                    None if args.min_calibration_precision is None else float(args.min_calibration_precision)
                ),
            },
            f,
            indent=2,
        )
    print(f"Threshold saved to '{args.threshold_output_path}'")

    print("\n--> Evaluating model performance on the test set...")
    if lower_scores_are_more_anomalous:
        y_pred_mapped = np.where(test_scores < threshold, 1, 0)
    else:
        y_pred_mapped = np.where(test_scores >= threshold, 1, 0)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred_mapped, target_names=["Normal", "Anomaly"]))

    end_time = time.time()
    print(f"\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()
