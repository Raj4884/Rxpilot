"""
RxPilot — Evaluation runner.

Phase 1: Placeholder structure. Phase 2 will add:
  - Field-level precision/recall/F1 for extraction
  - Safety recall on adversarial set
  - Validation precision on known-good/bad records
  - Baseline score tracking in checked-in JSON
  - GitHub Actions CI gate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GOLDEN_SET_DIR = Path(__file__).parent / "golden_set"
BASELINE_FILE = Path(__file__).parent / "baseline_scores.json"


def load_golden_set() -> list[dict]:
    """Load golden test cases from the golden_set directory."""
    cases = []
    for expected_file in sorted(GOLDEN_SET_DIR.glob("*_expected.json")):
        with open(expected_file) as f:
            case = json.load(f)
            case["_file"] = expected_file.name
            cases.append(case)
    return cases


def run_extraction_eval() -> dict[str, float]:
    """
    Run extraction evaluation against the golden set.
    Returns per-field precision/recall/F1 scores.

    Phase 2: Will implement actual scoring logic.
    """
    cases = load_golden_set()
    if not cases:
        logger.warning("No golden set test cases found in %s", GOLDEN_SET_DIR)
        return {"status": "no_test_cases"}

    logger.info("Loaded %d golden set cases", len(cases))
    # Phase 2: Implement actual evaluation
    return {
        "status": "placeholder",
        "test_cases": len(cases),
        "message": "Extraction eval will be implemented in Phase 2",
    }


def run_safety_eval() -> dict[str, float]:
    """
    Run safety agent evaluation against the adversarial set.

    Phase 2: Will implement recall/false-positive scoring.
    """
    return {
        "status": "placeholder",
        "message": "Safety eval will be implemented in Phase 2",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("RxPilot Evaluation Suite")
    print("=" * 60)

    print("\n[Extraction Eval]")
    extraction_results = run_extraction_eval()
    print(json.dumps(extraction_results, indent=2))

    print("\n[Safety Eval]")
    safety_results = run_safety_eval()
    print(json.dumps(safety_results, indent=2))

