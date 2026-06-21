"""
RxPilot — Evaluation metrics.

Phase 2 will implement:
  - field_precision_recall_f1: per-field extraction accuracy
  - safety_recall: drug interaction detection rate
  - validation_precision: anomaly detection accuracy
"""

from __future__ import annotations


def field_precision_recall_f1(
    predicted: dict,
    expected: dict,
    field_name: str,
) -> dict[str, float]:
    """
    Calculate precision, recall, and F1 for a single field.

    Phase 2: Will handle fuzzy matching for medicine names,
    date normalization for dates, numeric tolerance for prices.
    """
    # Placeholder — Phase 2
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def safety_recall(
    flagged_interactions: list[tuple[str, str]],
    expected_interactions: list[tuple[str, str]],
) -> dict[str, float]:
    """
    Calculate recall and false-positive rate for safety agent.

    Phase 2: Will implement actual scoring.
    """
    # Placeholder — Phase 2
    return {"recall": 0.0, "false_positive_rate": 0.0}
