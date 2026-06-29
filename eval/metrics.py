"""
RxPilot — Evaluation Metrics (Phase 2).

Implements field-level precision/recall/F1 for extraction evaluation
and recall/precision for safety agent evaluation.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


# ──────────────────────────────────────────────
# String normalization helpers
# ──────────────────────────────────────────────


def _normalize_str(s: str | None) -> str:
    """Lowercase, strip whitespace."""
    if s is None:
        return ""
    return str(s).lower().strip()


def _normalize_medicine_name(name: str) -> str:
    """
    Strip dosage and dosage-form suffixes for matching.
    'Paracetamol 500mg' -> 'paracetamol'
    """
    import re
    name = name.lower().strip()
    name = re.sub(r'\d+\s*(mg|ml|mcg|g|iu|units?)\b', '', name)
    name = re.sub(
        r'\b(tablets?|capsules?|syrup|injection|cream|ointment|drops?|'
        r'suspension|solution|gel|patch|inhaler|spray|powder|sachets?)\b',
        '', name
    )
    return re.sub(r'\s+', ' ', name).strip()


def _string_similarity(a: str, b: str) -> float:
    """Levenshtein-ish similarity using SequenceMatcher (0-1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _names_match(pred: str, expected: str, threshold: float = 0.7) -> bool:
    """True if medicine names are similar enough."""
    pn = _normalize_medicine_name(pred)
    en = _normalize_medicine_name(expected)
    if pn == en:
        return True
    sim = _string_similarity(pn, en)
    return sim >= threshold


# ──────────────────────────────────────────────
# Item matching
# ──────────────────────────────────────────────


def match_items(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """
    Match predicted items to expected items by medicine name.

    Returns a list of (predicted, expected) pairs.
    Unmatched predicted items are paired with None expected.
    Unmatched expected items are paired with None predicted.
    """
    matched: list[tuple[dict | None, dict | None]] = []
    used_expected = set()

    for pred in predicted:
        pred_name = pred.get("medicine_name", "")
        best_match = None
        best_sim = 0.0
        best_idx = -1

        for i, exp in enumerate(expected):
            if i in used_expected:
                continue
            exp_name = exp.get("medicine_name", "")
            if _names_match(pred_name, exp_name):
                sim = _string_similarity(
                    _normalize_medicine_name(pred_name),
                    _normalize_medicine_name(exp_name),
                )
                if sim > best_sim:
                    best_sim = sim
                    best_match = exp
                    best_idx = i

        if best_match is not None:
            matched.append((pred, best_match))
            used_expected.add(best_idx)
        else:
            matched.append((pred, None))  # False positive

    # Unmatched expected = false negatives
    for i, exp in enumerate(expected):
        if i not in used_expected:
            matched.append((None, exp))

    return matched


# ──────────────────────────────────────────────
# Field-level scoring
# ──────────────────────────────────────────────

SCORED_FIELDS = [
    "medicine_name",
    "batch_number",
    "expiry_date",
    "manufacture_date",
    "quantity",
    "price",
    "supplier_name",
]


def _field_correct(pred_val: Any, exp_val: Any, field: str) -> bool:
    """Check if a single field is correctly predicted."""
    # Both None/missing = correct
    if pred_val is None and exp_val is None:
        return True
    if pred_val is None or exp_val is None:
        return False

    if field == "medicine_name":
        return _names_match(str(pred_val), str(exp_val))

    if field in ("quantity",):
        try:
            return int(pred_val) == int(exp_val)
        except (ValueError, TypeError):
            return False

    if field == "price":
        try:
            # Allow 1% tolerance for price
            p, e = float(pred_val), float(exp_val)
            if e == 0:
                return p == 0
            return abs(p - e) / e < 0.01
        except (ValueError, TypeError):
            return False

    # String fields — exact match after normalization
    return _normalize_str(str(pred_val)) == _normalize_str(str(exp_val))


def field_precision_recall_f1(
    predicted_items: list[dict[str, Any]],
    expected_items: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Compute per-field precision, recall, and F1 scores.

    For each field, counts:
      - TP: predicted correctly (non-null prediction matches expected)
      - FP: predicted non-null but wrong (or extra item)
      - FN: expected non-null but not predicted (or missing item)

    Returns dict: { field_name: { precision, recall, f1, tp, fp, fn } }
    """
    pairs = match_items(predicted_items, expected_items)

    field_counts: dict[str, dict[str, int]] = {
        f: {"tp": 0, "fp": 0, "fn": 0} for f in SCORED_FIELDS
    }

    for pred, exp in pairs:
        for field in SCORED_FIELDS:
            pred_val = pred.get(field) if pred else None
            exp_val = exp.get(field) if exp else None

            if exp_val is not None:
                # Expected has a value
                if pred_val is not None and _field_correct(pred_val, exp_val, field):
                    field_counts[field]["tp"] += 1
                elif pred_val is not None:
                    field_counts[field]["fp"] += 1
                    field_counts[field]["fn"] += 1
                else:
                    field_counts[field]["fn"] += 1
            elif pred_val is not None:
                # Predicted something when nothing expected
                field_counts[field]["fp"] += 1

    results: dict[str, dict[str, float]] = {}
    for field, counts in field_counts.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[field] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return results


def aggregate_scores(
    per_field_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute macro-averaged precision, recall, F1 across all fields."""
    fields = list(per_field_scores.values())
    if not fields:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    avg_precision = sum(f["precision"] for f in fields) / len(fields)
    avg_recall = sum(f["recall"] for f in fields) / len(fields)
    avg_f1 = sum(f["f1"] for f in fields) / len(fields)

    return {
        "macro_precision": round(avg_precision, 4),
        "macro_recall": round(avg_recall, 4),
        "macro_f1": round(avg_f1, 4),
    }


# ──────────────────────────────────────────────
# Item-level scoring
# ──────────────────────────────────────────────


def item_level_scores(
    predicted_items: list[dict[str, Any]],
    expected_items: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Item-level precision and recall based on medicine name matching.
    """
    pairs = match_items(predicted_items, expected_items)
    tp = sum(1 for p, e in pairs if p is not None and e is not None)
    fp = sum(1 for p, e in pairs if p is not None and e is None)
    fn = sum(1 for p, e in pairs if p is None and e is not None)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "item_precision": round(precision, 4),
        "item_recall": round(recall, 4),
        "item_f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ──────────────────────────────────────────────
# Safety eval scoring
# ──────────────────────────────────────────────


def _normalize_drug_pair(pair: list[str] | tuple[str, str]) -> frozenset[str]:
    """Normalize a drug pair to a frozenset for comparison."""
    return frozenset(_normalize_medicine_name(d) for d in pair)


def safety_recall(
    predicted_flags: list[dict[str, Any]],
    expected_flags: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Compute safety agent recall and precision.

    Matching: drug pairs are order-independent, name-fuzzy.

    Args:
        predicted_flags: list of { drug_pair: [drug_a, drug_b], severity: ... }
        expected_flags: list of { drug_pair: [drug_a, drug_b], severity: ... }

    Returns:
        { recall, precision, f1, detected, total_expected, false_positives }
    """
    expected_pairs = [
        _normalize_drug_pair(f.get("drug_pair", []))
        for f in expected_flags
    ]
    predicted_pairs = [
        _normalize_drug_pair(f.get("drug_pair", []))
        for f in predicted_flags
    ]

    detected = sum(
        1 for ep in expected_pairs
        if any(_pair_matches(ep, pp) for pp in predicted_pairs)
    )
    false_positives = sum(
        1 for pp in predicted_pairs
        if not any(_pair_matches(ep, pp) for ep in expected_pairs)
    )

    recall = detected / len(expected_pairs) if expected_pairs else 1.0
    precision = (
        len(predicted_pairs) - false_positives
    ) / len(predicted_pairs) if predicted_pairs else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "detected": detected,
        "total_expected": len(expected_pairs),
        "false_positives": false_positives,
    }


def _pair_matches(a: frozenset[str], b: frozenset[str]) -> bool:
    """Check if two drug pairs match (fuzzy name matching)."""
    if a == b:
        return True
    # Try fuzzy matching each element
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != 2 or len(b_list) != 2:
        return False
    # Try both orderings
    return (
        _string_similarity(a_list[0], b_list[0]) > 0.7
        and _string_similarity(a_list[1], b_list[1]) > 0.7
    ) or (
        _string_similarity(a_list[0], b_list[1]) > 0.7
        and _string_similarity(a_list[1], b_list[0]) > 0.7
    )
