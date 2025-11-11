"""Tests for the HDFS sequence builder script."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from build_sequences import (  # type: ignore
    build_sequences,
    load_labels,
    save_outputs,
    stream_structured_events,
)


class BuildSequencesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def create_structured_csv(self) -> Path:
        rows = [
            {
                "timestamp": 1,
                "pid": 1,
                "severity": "INFO",
                "component": "dfs.DataNode",
                "message": "Receiving block blk_-1 src: /n1 dest: /n2",
                "EventId": "E1",
            },
            {
                "timestamp": 2,
                "pid": 2,
                "severity": "INFO",
                "component": "dfs.DataNode",
                "message": "Receiving block blk_-1 src: /n2 dest: /n3",
                "EventId": "E2",
            },
            {
                "timestamp": 3,
                "pid": 3,
                "severity": "INFO",
                "component": "dfs.FSNamesystem",
                "message": "BLOCK* NameSystem.allocateBlock: /file blk_-2",
                "EventId": "E3",
            },
        ]
        path = self.tmp_path / "structured.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def create_labels_csv(self) -> Path:
        rows = [
            {"BlockId": "blk_-1", "Label": "Normal"},
            {"BlockId": "blk_-2", "Label": "Anomaly"},
        ]
        path = self.tmp_path / "labels.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_pipeline_generates_sequences_and_labels(self) -> None:
        structured = self.create_structured_csv()
        labels_path = self.create_labels_csv()

        block_events = stream_structured_events(
            structured,
            message_column="message",
            event_column="EventId",
            chunk_size=2,
        )
        labels_df = load_labels(labels_path)
        result = build_sequences(block_events, labels_df)

        output_npz = self.tmp_path / "sequences.npz"
        block_index_csv = self.tmp_path / "block_index.csv"
        save_outputs(result, output_npz=output_npz, block_index_csv=block_index_csv)

        with np.load(output_npz, allow_pickle=True) as data:
            x_data = data["x_data"]
            y_data = data["y_data"]
        self.assertEqual(len(x_data), 2)
        self.assertListEqual(list(x_data[0]), ["E1", "E2"])
        self.assertListEqual(list(x_data[1]), ["E3"])
        self.assertTrue((y_data == np.array([0, 1])).all())

        block_index = pd.read_csv(block_index_csv, index_col="Index")
        self.assertListEqual(block_index["BlockId"].tolist(), ["blk_-1", "blk_-2"])
        self.assertListEqual(block_index["Label"].tolist(), [0, 1])

    def test_missing_blocks_reported(self) -> None:
        structured = self.create_structured_csv()
        labels_path = self.tmp_path / "labels.csv"
        pd.DataFrame([{"BlockId": "blk_missing", "Label": "Normal"}]).to_csv(labels_path, index=False)

        block_events = stream_structured_events(structured, "message", "EventId", chunk_size=2)
        labels_df = load_labels(labels_path)
        result = build_sequences(block_events, labels_df)

        self.assertEqual(result.block_ids, [])
        self.assertEqual(result.sequences, [])
        self.assertEqual(result.labels, [])
        self.assertIn("blk_missing", result.missing_blocks)


if __name__ == "__main__":
    unittest.main()
