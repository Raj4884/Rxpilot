"""
RxPilot — Tests for the Safety RAG Agent and Metrics (Phase 2).

Tests:
  - Drug name normalization
  - CSV-based interaction search (no DB required)
  - Safety agent: single drug no flags
  - Safety agent: known interaction pairs detected
  - Safety agent: graceful degradation without Anthropic key
  - Metrics: safety_recall P/R/F1 computation
  - Metrics: field_precision_recall_f1
  - Metrics: medicine name matching
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.state import ExtractedBillFields, ExtractionResult, PharmacyState


# ── Fixtures ──


def _make_state_with_meds(med_names: list[str]) -> PharmacyState:
    """Create PharmacyState with given medicine names."""
    items = [ExtractedBillFields(medicine_name=name) for name in med_names]
    extraction = ExtractionResult(items=items, raw_llm_output="[]")
    return PharmacyState(
        input_type="image",
        raw_input_ref="/tmp/test.jpg",
        extracted_fields=extraction,
    )


# ── Tests: Drug Name Normalization ──


def test_normalize_strips_dosage():
    """Normalization should strip dosage info."""
    from agents.safety_agent import _normalize_drug_name
    assert _normalize_drug_name("Paracetamol 500mg") == "paracetamol"
    assert _normalize_drug_name("Amoxicillin 250mg Capsules") == "amoxicillin"
    assert _normalize_drug_name("Warfarin 5mg Tablets") == "warfarin"


def test_normalize_strips_dosage_form():
    """Normalization should strip dosage form words."""
    from agents.safety_agent import _normalize_drug_name
    assert _normalize_drug_name("Metformin 500mg tablets") == "metformin"
    assert _normalize_drug_name("Omeprazole 20mg capsules") == "omeprazole"
    assert _normalize_drug_name("Insulin injection 100IU") == "insulin"


def test_normalize_lowercases():
    """Normalization should lowercase."""
    from agents.safety_agent import _normalize_drug_name
    assert _normalize_drug_name("WARFARIN") == "warfarin"
    assert _normalize_drug_name("Aspirin") == "aspirin"


# ── Tests: CSV Fallback Search ──


def test_csv_search_warfarin_aspirin():
    """Warfarin + Aspirin should be found in CSV as high-severity interaction."""
    from rag import _search_csv_fallback
    results = _search_csv_fallback(["warfarin", "aspirin"])
    assert len(results) > 0
    severities = [r["severity"] for r in results]
    assert "high" in severities


def test_csv_search_single_drug_no_results():
    """Single drug name should return no interactions."""
    from rag import _search_csv_fallback
    results = _search_csv_fallback(["paracetamol"])
    assert results == []


def test_csv_search_unrelated_drugs_no_results():
    """Two drugs with no known interaction should return empty."""
    from rag import _search_csv_fallback
    results = _search_csv_fallback(["vitamin d3", "cetirizine"])
    assert results == []


def test_csv_search_ciprofloxacin_theophylline():
    """Ciprofloxacin + Theophylline is a high-severity interaction."""
    from rag import _search_csv_fallback
    results = _search_csv_fallback(["ciprofloxacin", "theophylline"])
    assert len(results) > 0
    assert any(r["severity"] in ("high", "critical") for r in results)


def test_csv_search_paracetamol_warfarin():
    """Paracetamol + Warfarin should have a low-severity interaction in CSV."""
    from rag import _search_csv_fallback
    results = _search_csv_fallback(["paracetamol", "warfarin"])
    assert len(results) > 0


# ── Tests: Safety Agent ──


@pytest.mark.asyncio
async def test_safety_agent_single_drug_no_flags():
    """A bill with only one drug should produce no safety flags."""
    from agents.safety_agent import run_safety_check

    state = _make_state_with_meds(["Paracetamol 500mg"])
    result = await run_safety_check(state)
    assert result.safety_flags == []


@pytest.mark.asyncio
async def test_safety_agent_empty_items_no_flags():
    """Empty extraction should produce no safety flags."""
    from agents.safety_agent import run_safety_check

    state = PharmacyState(input_type="image", raw_input_ref="/tmp/test.jpg")
    result = await run_safety_check(state)
    assert result.safety_flags == []


@pytest.mark.asyncio
async def test_safety_agent_warfarin_aspirin_flagged():
    """Warfarin + Aspirin bill should produce a high-severity safety flag."""
    from agents.safety_agent import run_safety_check

    state = _make_state_with_meds(["Warfarin 5mg", "Aspirin 75mg"])

    # Mock LLM to avoid API call
    with patch("agents.safety_agent._llm_safety_assessment", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda interactions, drugs: interactions
        with patch("agents.safety_agent.search_interactions") as mock_search:
            mock_search.return_value = [
                {
                    "drug_a": "Warfarin",
                    "drug_b": "Aspirin",
                    "severity": "high",
                    "description": "Concurrent use significantly increases bleeding risk",
                    "mechanism": "Both inhibit hemostasis",
                    "source": "OpenFDA label section 7.1",
                }
            ]
            result = await run_safety_check(state)

    assert len(result.safety_flags) == 1
    flag = result.safety_flags[0]
    assert flag.severity == "high"
    assert "Warfarin" in flag.drug_pair or "warfarin" in str(flag.drug_pair).lower()


@pytest.mark.asyncio
async def test_safety_agent_no_interaction_no_flags():
    """Drugs with no known interaction should produce no safety flags."""
    from agents.safety_agent import run_safety_check

    state = _make_state_with_meds(["Vitamin D3", "Calcium Carbonate"])

    with patch("agents.safety_agent.search_interactions", return_value=[]):
        result = await run_safety_check(state)

    assert result.safety_flags == []


@pytest.mark.asyncio
async def test_safety_agent_multiple_interactions():
    """Multiple interactions should produce multiple flags."""
    from agents.safety_agent import run_safety_check

    state = _make_state_with_meds(["Ciprofloxacin 500mg", "Warfarin 5mg", "Theophylline 200mg"])

    mock_interactions = [
        {
            "drug_a": "Ciprofloxacin",
            "drug_b": "Theophylline",
            "severity": "high",
            "description": "Ciprofloxacin inhibits theophylline metabolism",
            "mechanism": "CYP1A2 inhibition",
            "source": "OpenFDA label",
        },
        {
            "drug_a": "Ciprofloxacin",
            "drug_b": "Warfarin",
            "severity": "moderate",
            "description": "Ciprofloxacin may enhance anticoagulant effect",
            "mechanism": "CYP1A2 inhibition",
            "source": "FDA drug interaction guidance",
        },
    ]

    with patch("agents.safety_agent._llm_safety_assessment", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda interactions, drugs: interactions
        with patch("agents.safety_agent.search_interactions", return_value=mock_interactions):
            result = await run_safety_check(state)

    assert len(result.safety_flags) == 2
    severities = {f.severity for f in result.safety_flags}
    assert "high" in severities
    assert "moderate" in severities


@pytest.mark.asyncio
async def test_safety_agent_bad_severity_defaults_to_moderate():
    """Unknown severity value should default to 'moderate'."""
    from agents.safety_agent import run_safety_check

    state = _make_state_with_meds(["Drug A", "Drug B"])

    mock_interactions = [{
        "drug_a": "Drug A",
        "drug_b": "Drug B",
        "severity": "unknown_value",  # Invalid
        "description": "Some interaction",
        "mechanism": "",
        "source": "Test",
    }]

    with patch("agents.safety_agent._llm_safety_assessment", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda interactions, drugs: interactions
        with patch("agents.safety_agent.search_interactions", return_value=mock_interactions):
            result = await run_safety_check(state)

    assert len(result.safety_flags) == 1
    assert result.safety_flags[0].severity == "moderate"


# ── Tests: Eval Metrics ──


def test_field_precision_recall_perfect_prediction():
    """Perfect prediction should give precision=recall=f1=1.0."""
    from eval.metrics import field_precision_recall_f1

    items = [{"medicine_name": "Warfarin", "batch_number": "WAR-001",
              "expiry_date": "2026-12-31", "manufacture_date": "2024-01-01",
              "quantity": 30, "price": 120.0, "supplier_name": "Cipla"}]

    scores = field_precision_recall_f1(items, items)
    for field, s in scores.items():
        assert s["precision"] == 1.0, f"{field} precision should be 1.0"
        assert s["recall"] == 1.0, f"{field} recall should be 1.0"
        assert s["f1"] == 1.0, f"{field} F1 should be 1.0"


def test_field_recall_missing_items():
    """Missing all predicted items gives recall=0."""
    from eval.metrics import field_precision_recall_f1

    expected = [{"medicine_name": "Warfarin", "batch_number": "WAR-001",
                 "quantity": 30, "price": 120.0, "expiry_date": "2026-12-31",
                 "manufacture_date": "2024-01-01", "supplier_name": "Cipla"}]

    scores = field_precision_recall_f1([], expected)
    # With no predictions: all FN, so recall=0
    for field, s in scores.items():
        if expected[0].get(field) is not None:
            assert s["recall"] == 0.0, f"{field} recall should be 0 with no predictions"


def test_safety_recall_all_detected():
    """All expected interactions detected should give recall=1."""
    from eval.metrics import safety_recall

    expected = [{"drug_pair": ["Warfarin", "Aspirin"], "severity": "high"}]
    predicted = [{"drug_pair": ["Warfarin", "Aspirin"], "severity": "high"}]

    scores = safety_recall(predicted, expected)
    assert scores["recall"] == 1.0


def test_safety_recall_none_detected():
    """No interactions detected should give recall=0."""
    from eval.metrics import safety_recall

    expected = [{"drug_pair": ["Warfarin", "Aspirin"], "severity": "high"}]
    predicted = []

    scores = safety_recall(predicted, expected)
    assert scores["recall"] == 0.0


def test_safety_recall_order_independent():
    """Drug pair order should not matter for matching."""
    from eval.metrics import safety_recall

    expected = [{"drug_pair": ["Warfarin", "Aspirin"], "severity": "high"}]
    predicted = [{"drug_pair": ["Aspirin", "Warfarin"], "severity": "high"}]  # Reversed

    scores = safety_recall(predicted, expected)
    assert scores["recall"] == 1.0


def test_safety_recall_false_positives():
    """Extra predicted interactions not in expected should be false positives."""
    from eval.metrics import safety_recall

    expected = [{"drug_pair": ["Warfarin", "Aspirin"], "severity": "high"}]
    predicted = [
        {"drug_pair": ["Warfarin", "Aspirin"], "severity": "high"},
        {"drug_pair": ["Metformin", "Insulin"], "severity": "low"},  # FP
    ]

    scores = safety_recall(predicted, expected)
    assert scores["recall"] == 1.0
    assert scores["false_positives"] == 1


def test_safety_recall_no_expected_no_predicted():
    """No expected and no predicted should give recall=1 (vacuously true)."""
    from eval.metrics import safety_recall

    scores = safety_recall([], [])
    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0


def test_medicine_name_fuzzy_matching():
    """Fuzzy medicine name matching should handle dosage differences."""
    from eval.metrics import _names_match

    assert _names_match("Paracetamol 500mg", "Paracetamol")
    assert _names_match("Amoxicillin 250mg Capsules", "Amoxicillin")
    assert not _names_match("Metformin", "Metoprolol")  # Different drugs


# ── Tests: Eval Runner ──


def test_eval_runner_loads_golden_set():
    """Eval runner should load all 5 golden set cases."""
    from eval.run_eval import load_golden_set
    cases = load_golden_set()
    assert len(cases) >= 5


def test_eval_runner_extraction_eval():
    """Extraction eval should return completed status with non-zero scores."""
    from eval.run_eval import load_golden_set, run_extraction_eval
    cases = load_golden_set()
    results = run_extraction_eval(cases)
    assert results["status"] == "completed"
    assert results["cases_evaluated"] >= 5
    assert "macro" in results
    assert results["macro"]["macro_f1"] > 0


def test_eval_runner_safety_eval():
    """Safety eval should return completed status."""
    from eval.run_eval import load_golden_set, run_safety_eval
    cases = load_golden_set()
    results = run_safety_eval(cases)
    assert results["status"] in ("completed", "no_cases_with_flags")
