"""Tests for the Drain-based template mining script."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from mine_templates import mine_templates, setup_logging


class MineTemplatesTests(unittest.TestCase):
    """Integration-style checks for the template mining helper."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        setup_logging("ERROR")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def make_args(self, **overrides) -> argparse.Namespace:
        defaults = {
            "input_csv": str(self.tmp_path / "input.csv"),
            "output_structured": str(self.tmp_path / "structured.csv"),
            "templates_json": str(self.tmp_path / "templates.json"),
            "event_map_json": str(self.tmp_path / "event_map.json"),
            "message_column": "message",
            "config_file": None,
            "parser_depth": 4,
            "parser_max_children": 100,
            "parser_sim_th": 0.5,
            "parser_extra_delimiters": "_",
            "log_level": "ERROR",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_mine_templates_generates_expected_outputs(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "timestamp": 1,
                    "pid": 111,
                    "severity": "INFO",
                    "component": "dfs.DataNode",
                    "message": "Receiving block blk_-1 src: /n1 dest: /n2",
                },
                {
                    "timestamp": 2,
                    "pid": 112,
                    "severity": "INFO",
                    "component": "dfs.DataNode",
                    "message": "Receiving block blk_-2 src: /n1 dest: /n3",
                },
                {
                    "timestamp": 3,
                    "pid": 200,
                    "severity": "INFO",
                    "component": "dfs.FSNamesystem",
                    "message": "BLOCK* NameSystem.allocateBlock: /file blk_-3",
                },
            ]
        )
        args = self.make_args()
        df.to_csv(args.input_csv, index=False)

        mine_templates(args)

        structured_path = Path(args.output_structured)
        self.assertTrue(structured_path.exists())
        structured_df = pd.read_csv(structured_path)
        self.assertIn("EventId", structured_df.columns)
        self.assertIn("EventTemplate", structured_df.columns)
        self.assertEqual(len(structured_df), len(df))
        unique_ids = sorted(structured_df["EventId"].unique(), key=lambda eid: int(eid[1:]))
        self.assertEqual(unique_ids, [f"E{i+1}" for i in range(len(unique_ids))])

        map_path = Path(args.event_map_json)
        self.assertTrue(map_path.exists())
        with map_path.open() as fh:
            event_map = json.load(fh)
        self.assertEqual(event_map["metadata"]["total_lines"], len(df))
        templates = event_map["templates"]
        self.assertGreaterEqual(len(templates), 2)
        self.assertTrue(all(template["event_id"].startswith("E") for template in templates))

        templates_state_path = Path(args.templates_json)
        self.assertTrue(templates_state_path.exists())

    def test_mine_templates_missing_message_column_raises(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "timestamp": 1,
                    "pid": 111,
                    "severity": "INFO",
                    "component": "dfs.DataNode",
                    "text": "Receiving block blk_-1 src: /n1 dest: /n2",
                }
            ]
        )
        args = self.make_args()
        df.to_csv(args.input_csv, index=False)
        with self.assertRaises(KeyError):
            mine_templates(args)

    def test_mine_templates_missing_input_file_raises(self) -> None:
        args = self.make_args()
        missing_input = Path(args.input_csv)
        self.assertFalse(missing_input.exists())
        with self.assertRaises(FileNotFoundError):
            mine_templates(args)


if __name__ == "__main__":
    unittest.main()
