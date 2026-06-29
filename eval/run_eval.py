"""
RxPilot — Evaluation Runner (Phase 2).

Runs extraction and safety evaluations against the golden set.

Extraction eval:
  - Loads expected items from golden_set/*.json
  - Compares against predicted items using field-level P/R/F1
  - Reports per-field and macro scores
  - Checks against baseline to detect regressions

Safety eval:
  - For golden cases with expected_safety_flags, checks that the
    CSV-based fallback search detects the known interactions
  - Reports recall and false positive rate

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --baseline  # Update baseline_scores.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from eval.metrics import (
    field_precision_recall_f1,
    aggregate_scores,
    item_level_scores,
    safety_recall,
    _normalize_medicine_name,
)

logger = logging.getLogger(__name__)

GOLDEN_SET_DIR = Path(__file__).parent / "golden_set"
BASELINE_FILE = Path(__file__).parent / "baseline_scores.json"


def load_golden_set() -> list[dict]:
    """Load all golden set cases."""
    cases = []
    for expected_file in sorted(GOLDEN_SET_DIR.glob("*_expected.json")):
        with open(expected_file, encoding="utf-8") as f:
            case = json.load(f)
            case["_file"] = expected_file.name
            cases.append(case)
    return cases


def run_extraction_eval(cases: list[dict]) -> dict:
    """
    Evaluate extraction quality against golden set expected items.

    For Phase 2, we evaluate using the expected items from each golden
    case as both 'predicted' and 'expected' to establish a baseline
    (since we don't have actual model predictions without running Claude).

    In CI, this would run actual extractions. For now, it validates the
    metrics code is wired correctly and produces sensible outputs.
    """
    all_per_field: list[dict] = []
    all_item_scores: list[dict] = []
    case_results = []

    for case in cases:
        expected_items = case.get("items", [])
        if not expected_items:
            continue

        # For now: use expected as predicted to validate metrics pipeline
        # In Phase 3 CI, actual Claude predictions will be injected here
        predicted_items = expected_items  # Perfect prediction baseline

        per_field = field_precision_recall_f1(predicted_items, expected_items)
        item_scores = item_level_scores(predicted_items, expected_items)

        case_results.append({
            "case": case["_file"],
            "description": case.get("description", ""),
            "items_expected": len(expected_items),
            "items_predicted": len(predicted_items),
            "item_f1": item_scores["item_f1"],
            "macro_f1": aggregate_scores(per_field)["macro_f1"],
        })
        all_per_field.append(per_field)
        all_item_scores.append(item_scores)

    if not all_per_field:
        return {"status": "no_cases", "cases": []}

    # Aggregate across all cases
    agg_per_field: dict[str, dict] = {}
    from eval.metrics import SCORED_FIELDS
    for field in SCORED_FIELDS:
        total_tp = sum(pf.get(field, {}).get("tp", 0) for pf in all_per_field)
        total_fp = sum(pf.get(field, {}).get("fp", 0) for pf in all_per_field)
        total_fn = sum(pf.get(field, {}).get("fn", 0) for pf in all_per_field)
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        agg_per_field[field] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
        }

    macro = aggregate_scores(agg_per_field)
    avg_item_f1 = sum(s["item_f1"] for s in all_item_scores) / len(all_item_scores)

    return {
        "status": "completed",
        "cases_evaluated": len(case_results),
        "macro": macro,
        "avg_item_f1": round(avg_item_f1, 4),
        "per_field": agg_per_field,
        "case_results": case_results,
    }


def run_safety_eval(cases: list[dict]) -> dict:
    """
    Evaluate safety RAG detection against golden set expected_safety_flags.

    Uses the CSV fallback search (no DB required) to check if known
    interactions are detected for drug combinations in each golden case.
    """
    # Import here to avoid heavy dep at module load
    try:
        from rag import _search_csv_fallback
    except ImportError as e:
        return {"status": "error", "message": f"RAG import failed: {e}"}

    all_expected: list[dict] = []
    all_predicted: list[dict] = []
    case_results = []

    for case in cases:
        expected_flags = case.get("expected_safety_flags", [])
        items = case.get("items", [])
        if not items:
            continue

        drug_names = [_normalize_medicine_name(i["medicine_name"]) for i in items]

        # Run CSV fallback search
        found_interactions = _search_csv_fallback(drug_names)
        predicted_flags = [
            {"drug_pair": [r["drug_a"], r["drug_b"]], "severity": r["severity"]}
            for r in found_interactions
        ]

        scores = safety_recall(predicted_flags, expected_flags)

        case_results.append({
            "case": case["_file"],
            "drugs_checked": drug_names,
            "expected_flags": len(expected_flags),
            "predicted_flags": len(predicted_flags),
            "recall": scores["recall"],
            "precision": scores["precision"],
            "f1": scores["f1"],
        })

        all_expected.extend(expected_flags)
        all_predicted.extend(predicted_flags)

    if not case_results:
        return {"status": "no_cases_with_flags"}

    overall = safety_recall(all_predicted, all_expected)

    return {
        "status": "completed",
        "cases_with_expected_flags": sum(
            1 for c in case_results if c["expected_flags"] > 0
        ),
        "overall_recall": overall["recall"],
        "overall_precision": overall["precision"],
        "overall_f1": overall["f1"],
        "detected": overall["detected"],
        "total_expected": overall["total_expected"],
        "false_positives": overall["false_positives"],
        "case_results": case_results,
    }


def check_regression(
    extraction_results: dict,
    safety_results: dict,
    baseline: dict,
) -> list[str]:
    """Check if scores regressed vs. baseline. Returns list of regression messages."""
    regressions = []

    if baseline.get("extraction"):
        base_f1 = baseline["extraction"].get("macro_f1", 0)
        curr_f1 = extraction_results.get("macro", {}).get("macro_f1", 0)
        if curr_f1 < base_f1 - 0.02:  # 2% tolerance
            regressions.append(
                f"Extraction macro F1 regressed: {curr_f1:.4f} < baseline {base_f1:.4f}"
            )

    if baseline.get("safety"):
        base_recall = baseline["safety"].get("overall_recall", 0)
        curr_recall = safety_results.get("overall_recall", 0)
        if curr_recall < base_recall - 0.05:  # 5% tolerance
            regressions.append(
                f"Safety recall regressed: {curr_recall:.4f} < baseline {base_recall:.4f}"
            )

    return regressions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    update_baseline = "--baseline" in sys.argv

    print("=" * 60)
    print("RxPilot Evaluation Suite (Phase 2)")
    print("=" * 60)

    cases = load_golden_set()
    print(f"\nLoaded {len(cases)} golden set cases from {GOLDEN_SET_DIR}")

    print("\n[Extraction Eval]")
    extraction_results = run_extraction_eval(cases)
    print(json.dumps(extraction_results, indent=2))

    print("\n[Safety Eval]")
    safety_results = run_safety_eval(cases)
    print(json.dumps(safety_results, indent=2))

    # Baseline comparison
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE, encoding="utf-8") as f:
            baseline = json.load(f)
        regressions = check_regression(extraction_results, safety_results, baseline)
        if regressions:
            print("\n[REGRESSIONS DETECTED]")
            for r in regressions:
                print(f"  FAIL: {r}")
            sys.exit(1)
        else:
            print("\n[Baseline Check] No regressions detected.")
    else:
        print("\n[Baseline] No baseline file found. Run with --baseline to create one.")

    # Update baseline
    if update_baseline:
        baseline_data = {
            "extraction": extraction_results.get("macro", {}),
            "safety": {
                "overall_recall": safety_results.get("overall_recall", 0),
                "overall_precision": safety_results.get("overall_precision", 0),
                "overall_f1": safety_results.get("overall_f1", 0),
            },
        }
        with open(BASELINE_FILE, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2)
        print(f"\n[Baseline] Updated baseline_scores.json")
