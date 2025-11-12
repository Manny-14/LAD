#!/usr/bin/env python3
"""Guardrail checks for LAD artifacts.

The helpers in this module validate that generated artifacts remain structurally
consistent across pipeline runs. They are intentionally lightweight so they can
run as part of every summary step or CI check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class GuardrailCheck:
    """Structured representation of a guardrail outcome."""

    name: str
    status: str
    details: dict

    def is_failure(self) -> bool:
        return self.status.lower() == "fail"


class GuardrailError(RuntimeError):
    """Raised when one or more guardrail checks fail."""


def _ensure_iterable_sequence(obj: object) -> bool:
    if isinstance(obj, (str, bytes)):
        return False
    try:
        iter(obj)  # type: ignore[arg-type]
    except TypeError:
        return False
    return True


def check_npz_schema(x_data: np.ndarray, y_data: Optional[np.ndarray]) -> GuardrailCheck:
    issues: List[str] = []
    if not isinstance(x_data, np.ndarray):
        issues.append("x_data is not a numpy.ndarray")
    if isinstance(x_data, np.ndarray):
        if x_data.dtype != object:
            issues.append(f"x_data dtype expected 'object', found '{x_data.dtype}'")
        if x_data.ndim != 1:
            issues.append(f"x_data expected to be 1-D, found ndim={x_data.ndim}")
        elif len(x_data) == 0:
            issues.append("x_data is empty")
        else:
            sample = x_data[0]
            if not _ensure_iterable_sequence(sample):
                issues.append("x_data entries must be iterable sequences of events")
    if y_data is not None:
        if not isinstance(y_data, np.ndarray):
            issues.append("y_data is not a numpy.ndarray")
        elif len(y_data) != len(x_data):
            issues.append(
                f"y_data length {len(y_data)} does not match x_data length {len(x_data)}"
            )

    status = "fail" if issues else "ok"
    details = {
        "num_sequences": int(len(x_data)),
        "has_labels": y_data is not None,
        "issues": issues,
    }
    return GuardrailCheck(name="npz_schema", status=status, details=details)


def check_block_index(block_index_path: Path, expected_rows: int) -> GuardrailCheck:
    if not block_index_path.exists():
        return GuardrailCheck(
            name="block_index",
            status="warn",
            details={
                "message": f"Block index not found at {block_index_path}; skipping length validation.",
                "expected_rows": expected_rows,
            },
        )

    df = pd.read_csv(block_index_path)
    issues: List[str] = []
    if "BlockId" not in df.columns:
        issues.append("BlockId column missing from block index")
    if len(df) != expected_rows:
        issues.append(
            f"Block index row count {len(df)} does not match expected {expected_rows}"
        )

    status = "fail" if issues else "ok"
    details = {
        "path": str(block_index_path),
        "rows": int(len(df)),
        "issues": issues,
    }
    return GuardrailCheck(name="block_index", status=status, details=details)


def _iter_unique_events(sequences: Iterable[Sequence[str]]) -> set[str]:
    unique: set[str] = set()
    for seq in sequences:
        if not _ensure_iterable_sequence(seq):
            continue
        for event in seq:
            unique.add(str(event))
    return unique


def check_vocabulary(
    sequences: np.ndarray,
    vocab_mapping: dict[str, int],
    *,
    drift_tolerance: float,
    max_unknown_events: int,
) -> GuardrailCheck:
    if not vocab_mapping:
        raise GuardrailError("Event vocabulary is empty; retrain the model.")

    vocab_events = set(vocab_mapping.keys())
    observed_events = _iter_unique_events(sequences)
    unknown_events = observed_events.difference(vocab_events)

    vocab_size = len(vocab_events)
    observed_size = len(observed_events)
    drift = 0.0 if vocab_size == 0 else abs(observed_size - vocab_size) / vocab_size

    issues: List[str] = []
    if drift > drift_tolerance:
        issues.append(
            f"Observed event count {observed_size} deviates from vocab size {vocab_size} by {drift:.2%}"
        )
    if len(unknown_events) > max_unknown_events:
        sample = sorted(list(unknown_events))[:10]
        issues.append(
            f"Encountered {len(unknown_events)} events not present in vocabulary (samples: {sample})"
        )

    status = "fail" if issues else "ok"
    details = {
        "vocab_size": vocab_size,
        "observed_event_count": observed_size,
        "drift": drift,
        "unknown_event_count": len(unknown_events),
    }
    if issues:
        details["issues"] = issues
    return GuardrailCheck(name="vocabulary", status=status, details=details)


def run_guardrails(
    *,
    sequences: np.ndarray,
    labels: Optional[np.ndarray],
    vocab_mapping: dict[str, int],
    block_index_path: Path,
    drift_tolerance: float,
    max_unknown_events: int,
) -> List[GuardrailCheck]:
    checks: List[GuardrailCheck] = []

    checks.append(check_npz_schema(sequences, labels))
    checks.append(check_block_index(block_index_path, expected_rows=len(sequences)))
    checks.append(
        check_vocabulary(
            sequences,
            vocab_mapping,
            drift_tolerance=drift_tolerance,
            max_unknown_events=max_unknown_events,
        )
    )

    failures = [check for check in checks if check.is_failure()]
    if failures:
        summary = ", ".join(f"{check.name}: {check.details.get('issues', 'unknown issue')}" for check in failures)
        raise GuardrailError(f"Guardrail checks failed: {summary}")

    return checks


__all__ = [
    "GuardrailCheck",
    "GuardrailError",
    "check_block_index",
    "check_npz_schema",
    "check_vocabulary",
    "run_guardrails",
]
