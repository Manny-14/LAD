"""Unit tests for guardrail helper functions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from guardrails import GuardrailError, check_vocabulary, run_guardrails


class GuardrailTests(unittest.TestCase):
    def _make_sequences(self, rows: list[list[str]]) -> np.ndarray:
        return np.array([np.array(seq, dtype=object) for seq in rows], dtype=object)

    def _write_block_index(self, tmpdir: Path, rows: int) -> Path:
        df = pd.DataFrame({"BlockId": [f"blk_{i}" for i in range(rows)]})
        path = tmpdir / "block_index.csv"
        df.to_csv(path, index=False)
        return path

    def test_run_guardrails_passes_with_clean_artifacts(self) -> None:
        sequences = self._make_sequences([["E1", "E2"], ["E2", "E3"], ["E1"]])
        vocab = {"E1": 0, "E2": 1, "E3": 2}
        labels = np.array([0, 1, 0], dtype=int)
        with tempfile.TemporaryDirectory() as tmp:
            block_index = self._write_block_index(Path(tmp), len(sequences))
            results = run_guardrails(
                sequences=sequences,
                labels=labels,
                vocab_mapping=vocab,
                block_index_path=block_index,
                drift_tolerance=0.5,
                max_unknown_events=0,
            )
        statuses = {check.name: check.status for check in results}
        self.assertEqual(statuses["npz_schema"], "ok")
        self.assertEqual(statuses["block_index"], "ok")
        self.assertEqual(statuses["vocabulary"], "ok")

    def test_check_vocabulary_detects_unknown_events(self) -> None:
        sequences = self._make_sequences([["E1"], ["E4"]])
        vocab = {"E1": 0, "E2": 1}
        with self.assertRaises(GuardrailError):
            run_guardrails(
                sequences=sequences,
                labels=None,
                vocab_mapping=vocab,
                block_index_path=Path("/nonexistent.csv"),
                drift_tolerance=0.5,
                max_unknown_events=0,
            )

    def test_check_vocabulary_detects_drift(self) -> None:
        sequences = self._make_sequences([["E1"], ["E2"]])
        vocab = {"E1": 0, "E2": 1, "E3": 2, "E4": 3}
        result = check_vocabulary(
            sequences,
            vocab,
            drift_tolerance=0.2,
            max_unknown_events=5,
        )
        self.assertEqual(result.status, "fail")
        self.assertIn("deviates", result.details.get("issues", [""])[0])


if __name__ == "__main__":
    unittest.main()
