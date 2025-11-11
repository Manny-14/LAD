#!/usr/bin/env python3
"""Build block-level event sequences from structured HDFS logs."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional

import numpy as np
import pandas as pd

BLOCK_ID_PATTERN = re.compile(r"(blk_[0-9\-]+)")
LOGGER = logging.getLogger("build_sequences")


@dataclass
class SequenceBuildResult:
    block_ids: List[str]
    sequences: List[List[str]]
    labels: Optional[List[int]]
    missing_blocks: List[str]
    unused_blocks: List[str]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert structured log templates into block-level event sequences compatible with training artifacts."
    )
    parser.add_argument(
        "--structured-csv",
        default="preprocessed/HDFS.log_structured.csv",
        help="CSV produced by mine_templates.py containing EventId assignments (default: preprocessed/HDFS.log_structured.csv)",
    )
    parser.add_argument(
        "--event-map-json",
        default="preprocessed/event_id_map.json",
        help="Optional JSON describing template metadata; used for validation/logging.",
    )
    parser.add_argument(
        "--labels-csv",
        default="data/anomaly_label.csv",
        help="Optional CSV with BlockId/Label columns to generate y_data. Leave empty to skip labels.",
    )
    parser.add_argument(
        "--output-npz",
        default="preprocessed/HDFS_sequences.npz",
        help="Destination NPZ file containing x_data (and y_data when labels are provided).",
    )
    parser.add_argument(
        "--block-index-csv",
        default="preprocessed/block_id_index.csv",
        help="CSV mapping each sequence row index to the originating BlockId.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Number of rows per chunk when streaming the structured CSV (default: 500000).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level (default: INFO).",
    )
    parser.add_argument(
        "--message-column",
        default="message",
        help="Column containing the raw log message used to extract block identifiers (default: message).",
    )
    parser.add_argument(
        "--event-column",
        default="EventId",
        help="Column containing the mined event identifiers (default: EventId).",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s | %(levelname)s | %(message)s")


def extract_block_ids(messages: pd.Series, message_column: str) -> pd.Series:
    block_ids = messages.str.extract(BLOCK_ID_PATTERN, expand=False)
    missing = block_ids.isna().sum()
    if missing:
        LOGGER.warning(
            "Encountered %s messages without a block identifier in column '%s'; they will be skipped.",
            missing,
            message_column,
        )
    return block_ids


def stream_structured_events(
    csv_path: Path,
    message_column: str,
    event_column: str,
    chunk_size: int,
) -> OrderedDict[str, List[str]]:
    """Stream the structured CSV and build an ordered mapping of block -> event sequence."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Structured CSV not found: {csv_path}")

    block_events: OrderedDict[str, List[str]] = OrderedDict()
    total_rows = 0
    LOGGER.info("Loading structured log from %s", csv_path)

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        total_rows += len(chunk)
        if message_column not in chunk.columns:
            raise KeyError(
                f"Column '{message_column}' not found in structured CSV. Available columns: {list(chunk.columns)}"
            )
        if event_column not in chunk.columns:
            raise KeyError(
                f"Column '{event_column}' not found in structured CSV. Available columns: {list(chunk.columns)}"
            )

        chunk = chunk[[message_column, event_column]].copy()
        chunk["BlockId"] = extract_block_ids(chunk[message_column], message_column)
        chunk.dropna(subset=["BlockId"], inplace=True)

        for block_id, event_id in zip(chunk["BlockId"], chunk[event_column]):
            if block_id not in block_events:
                block_events[block_id] = []
            block_events[block_id].append(str(event_id))

    LOGGER.info("Processed %s structured log rows", total_rows)
    LOGGER.info("Identified %s unique blocks", len(block_events))
    return block_events


def load_labels(labels_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if labels_path is None or not labels_path:
        return None
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")
    labels_df = pd.read_csv(labels_path)
    required = {"BlockId", "Label"}
    if not required.issubset(labels_df.columns):
        raise KeyError(
            f"Labels CSV must contain columns {required}; found {set(labels_df.columns)}"
        )
    return labels_df


def label_to_int(label: str) -> int:
    return 1 if str(label).strip().lower() == "anomaly" else 0


def build_sequences(
    block_events: MutableMapping[str, List[str]],
    labels_df: Optional[pd.DataFrame],
) -> SequenceBuildResult:
    block_ids: List[str] = []
    sequences: List[List[str]] = []
    labels: Optional[List[int]] = [] if labels_df is not None else None
    missing_blocks: List[str] = []

    if labels_df is not None:
        for row in labels_df.itertuples(index=False):
            block_id = str(row.BlockId)
            events = block_events.pop(block_id, None)
            if events is None:
                missing_blocks.append(block_id)
                continue
            block_ids.append(block_id)
            sequences.append(events)
            assert labels is not None  # for type checker
            labels.append(label_to_int(row.Label))
    else:
        for block_id, events in block_events.items():
            block_ids.append(block_id)
            sequences.append(events)

    unused_blocks = list(block_events.keys()) if labels_df is not None else []
    return SequenceBuildResult(block_ids, sequences, labels, missing_blocks, unused_blocks)


def save_outputs(
    result: SequenceBuildResult,
    output_npz: Path,
    block_index_csv: Path,
) -> None:
    block_index_csv.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    payload = {"x_data": np.array(result.sequences, dtype=object)}
    if result.labels is not None:
        payload["y_data"] = np.array(result.labels, dtype=int)

    np.savez(output_npz, **payload)
    LOGGER.info("Saved sequences to %s (x_data=%s)%s",
                output_npz,
                payload["x_data"].shape,
                " y_data included" if "y_data" in payload else "")

    block_index = pd.DataFrame({"BlockId": result.block_ids})
    if result.labels is not None:
        block_index["Label"] = result.labels
    block_index.to_csv(block_index_csv, index_label="Index")
    LOGGER.info("Wrote block index to %s", block_index_csv)


def log_summary(result: SequenceBuildResult) -> None:
    LOGGER.info("Total sequences produced: %s", len(result.sequences))
    if result.labels is not None:
        anomalies = sum(result.labels)
        LOGGER.info("Anomalies: %s | Normals: %s", anomalies, len(result.labels) - anomalies)
    if result.missing_blocks:
        LOGGER.warning("%s labeled blocks were not found in the structured log (examples: %s)",
                       len(result.missing_blocks), result.missing_blocks[:5])
    if result.unused_blocks:
        LOGGER.info("%s blocks from the structured log lacked labels and were excluded.",
                    len(result.unused_blocks))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    structured_path = Path(args.structured_csv)
    event_map_path = Path(args.event_map_json)
    labels_path = Path(args.labels_csv) if args.labels_csv else None
    output_npz = Path(args.output_npz)
    block_index_csv = Path(args.block_index_csv)

    if event_map_path.exists():
        with event_map_path.open() as fp:
            metadata = json.load(fp).get("metadata", {})
        LOGGER.info(
            "Event map metadata: %s templates, source=%s",
            metadata.get("total_templates"),
            metadata.get("source_csv"),
        )

    block_events = stream_structured_events(
        structured_path,
        message_column=args.message_column,
        event_column=args.event_column,
        chunk_size=args.chunk_size,
    )

    labels_df = load_labels(labels_path)
    result = build_sequences(block_events, labels_df)
    save_outputs(result, output_npz=output_npz, block_index_csv=block_index_csv)
    log_summary(result)


if __name__ == "__main__":
    main()
