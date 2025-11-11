"""Smoke tests for the Isolation Forest training CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class IsolationForestTrainingCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_tiny_sequences(self) -> Path:
        sequences = np.array(
            [
                np.array(["E1", "E2"], dtype=object),
                np.array(["E2", "E3"], dtype=object),
                np.array(["E1", "E3"], dtype=object),
                np.array(["E4"], dtype=object),
                np.array(["E4", "E2"], dtype=object),
                np.array(["E5"], dtype=object),
            ],
            dtype=object,
        )
        labels = np.array([0, 0, 0, 1, 1, 0], dtype=int)
        npz_path = self.tmp_path / "tiny_sequences.npz"
        np.savez(npz_path, x_data=sequences, y_data=labels)
        return npz_path

    def _write_block_index(self, rows: int) -> Path:
        df = pd.DataFrame(
            {
                "Index": list(range(rows)),
                "BlockId": [f"blk_{i}" for i in range(rows)],
                "Label": [0] * rows,
            }
        )
        path = self.tmp_path / "block_index.csv"
        df.to_csv(path, index=False)
        return path

    def _train_smoke_model(self) -> tuple[Path, Path, Path, Path]:
        data_path = self._write_tiny_sequences()
        model_path = self.tmp_path / "model.joblib"
        vocab_path = self.tmp_path / "vocab.json"
        threshold_path = self.tmp_path / "threshold.json"

        cmd = [
            sys.executable,
            "-m",
            "ml_models.isolation_forest_training",
            "--data-path",
            str(data_path),
            "--model-output-path",
            str(model_path),
            "--skip-grid-search",
            "--vocab-output-path",
            str(vocab_path),
            "--threshold-output-path",
            str(threshold_path),
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            self.fail(
                "Isolation Forest training CLI failed with return code "
                f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        self.assertTrue(model_path.exists(), "Trained model artifact was not created.")
        self.assertTrue(vocab_path.exists(), "Event vocabulary JSON was not created.")
        self.assertTrue(threshold_path.exists(), "Threshold JSON was not created.")

        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        self.assertIn("score_threshold", payload)
        self.assertIn("algorithm", payload)
        self.assertIn("score_direction", payload)

        return data_path, model_path, vocab_path, threshold_path

    def test_cli_trains_and_creates_model(self) -> None:
        _, model_path, vocab_path, threshold_path = self._train_smoke_model()
        self.assertTrue(model_path.exists())
        self.assertTrue(vocab_path.exists())
        self.assertTrue(threshold_path.exists())

    def test_inference_cli_generates_predictions(self) -> None:
        data_path, model_path, vocab_path, threshold_path = self._train_smoke_model()
        block_index_path = self._write_block_index(rows=6)
        output_path = self.tmp_path / "predictions.csv"

        cmd = [
            sys.executable,
            "-m",
            "ml_models.isolation_forest_inference",
            "--data-path",
            str(data_path),
            "--model-path",
            str(model_path),
            "--vocab-path",
            str(vocab_path),
            "--threshold-path",
            str(threshold_path),
            "--block-index-path",
            str(block_index_path),
            "--output-path",
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            self.fail(
                "Inference CLI failed with return code "
                f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        self.assertTrue(output_path.exists(), "Inference output CSV was not created.")
        predictions_df = pd.read_csv(output_path)
        self.assertIn("BlockId", predictions_df.columns)
        self.assertIn("prediction", predictions_df.columns)
        self.assertIn("anomaly_score", predictions_df.columns)
        self.assertIn("decision_threshold", predictions_df.columns)
        self.assertIn("score_direction", predictions_df.columns)
        self.assertIn("algorithm", predictions_df.columns)
        self.assertEqual(len(predictions_df), 6)


if __name__ == "__main__":
    unittest.main()
