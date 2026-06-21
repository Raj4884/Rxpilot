"""
Tests for agents.state — PharmacyState and sub-model validation.
"""

import pytest
from agents.state import (
    ExtractedBillFields,
    ExtractionResult,
    ForecastResult,
    PharmacyState,
    SafetyFlag,
)


class TestExtractedBillFields:
    """Tests for the ExtractedBillFields pydantic model."""

    def test_minimal_valid_item(self):
        """A medicine name is all that's required."""
        item = ExtractedBillFields(medicine_name="Paracetamol 500mg")
        assert item.medicine_name == "Paracetamol 500mg"
        assert item.batch_number is None
        assert item.price is None
        assert item.currency == "INR"

    def test_fully_populated_item(self):
        """All fields populated correctly."""
        item = ExtractedBillFields(
            medicine_name="Amoxicillin 250mg",
            batch_number="B2024-001",
            expiry_date="2026-03-15",
            manufacture_date="2024-03-15",
            quantity=100,
            unit="tablets",
            supplier_name="MedSupply Co.",
            price=45.50,
            currency="INR",
        )
        assert item.medicine_name == "Amoxicillin 250mg"
        assert item.batch_number == "B2024-001"
        assert item.expiry_date == "2026-03-15"
        assert item.quantity == 100
        assert item.price == 45.50

    def test_empty_medicine_name_rejected(self):
        """Medicine name cannot be empty."""
        with pytest.raises(Exception):
            ExtractedBillFields(medicine_name="")

    def test_negative_price_rejected(self):
        """Price must be >= 0."""
        with pytest.raises(Exception):
            ExtractedBillFields(medicine_name="Test", price=-10)

    def test_negative_quantity_rejected(self):
        """Quantity must be >= 0."""
        with pytest.raises(Exception):
            ExtractedBillFields(medicine_name="Test", quantity=-5)

    def test_date_normalization_dd_mm_yyyy(self):
        """DD/MM/YYYY format is normalized to YYYY-MM-DD."""
        item = ExtractedBillFields(
            medicine_name="Test",
            expiry_date="15/03/2026",
        )
        assert item.expiry_date == "2026-03-15"

    def test_date_normalization_dd_dash_mm_dash_yyyy(self):
        """DD-MM-YYYY format is normalized to YYYY-MM-DD."""
        item = ExtractedBillFields(
            medicine_name="Test",
            manufacture_date="01-06-2024",
        )
        assert item.manufacture_date == "2024-06-01"

    def test_date_normalization_mm_yyyy(self):
        """MM/YYYY format is normalized (day defaults to 01)."""
        item = ExtractedBillFields(
            medicine_name="Test",
            expiry_date="03/2026",
        )
        # Should normalize to a date string
        assert item.expiry_date is not None
        assert "2026" in item.expiry_date

    def test_null_dates_accepted(self):
        """None dates should be fine."""
        item = ExtractedBillFields(
            medicine_name="Test",
            expiry_date=None,
            manufacture_date=None,
        )
        assert item.expiry_date is None
        assert item.manufacture_date is None

    def test_expiry_after_manufacture_does_not_raise(self):
        """Expiry after manufacture is the normal case — should pass."""
        item = ExtractedBillFields(
            medicine_name="Test",
            expiry_date="2026-01-01",
            manufacture_date="2024-01-01",
        )
        assert item.expiry_date == "2026-01-01"

    def test_expiry_before_manufacture_does_not_raise(self):
        """
        Expiry before manufacture is an anomaly but the model doesn't raise —
        it's the validation agent's job to flag this.
        """
        item = ExtractedBillFields(
            medicine_name="Test",
            expiry_date="2023-01-01",
            manufacture_date="2024-01-01",
        )
        # Should construct without error
        assert item.expiry_date == "2023-01-01"


class TestExtractionResult:
    """Tests for ExtractionResult."""

    def test_empty_result(self):
        result = ExtractionResult()
        assert result.items == []
        assert result.raw_llm_output == ""
        assert result.parse_retries == 0

    def test_with_items(self):
        result = ExtractionResult(
            items=[
                ExtractedBillFields(medicine_name="Med A"),
                ExtractedBillFields(medicine_name="Med B", price=25.0),
            ],
            raw_llm_output='{"items": [...]}',
            parse_retries=1,
        )
        assert len(result.items) == 2
        assert result.parse_retries == 1


class TestSafetyFlag:
    """Tests for SafetyFlag."""

    def test_valid_safety_flag(self):
        flag = SafetyFlag(
            drug_pair=("Warfarin", "Aspirin"),
            severity="high",
            description="Increased bleeding risk",
            source="OpenFDA label section 7.1",
        )
        assert flag.drug_pair == ("Warfarin", "Aspirin")
        assert flag.severity == "high"

    def test_invalid_severity_rejected(self):
        with pytest.raises(Exception):
            SafetyFlag(
                drug_pair=("A", "B"),
                severity="extreme",  # not a valid literal
                description="test",
                source="test",
            )


class TestForecastResult:
    """Tests for ForecastResult."""

    def test_default_forecast(self):
        f = ForecastResult(medicine_name="Paracetamol")
        assert f.predicted_reorder_date is None
        assert f.confidence == 0.0

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            ForecastResult(medicine_name="Test", confidence=1.5)


class TestPharmacyState:
    """Tests for the main PharmacyState graph state."""

    def test_minimal_image_state(self):
        state = PharmacyState(
            input_type="image",
            raw_input_ref="/path/to/bill.jpg",
        )
        assert state.input_type == "image"
        assert state.trace_id  # auto-generated UUID
        assert state.extracted_fields is None
        assert state.validation_flags == []
        assert state.safety_flags == []

    def test_minimal_voice_state(self):
        state = PharmacyState(
            input_type="voice",
            raw_input_ref="/path/to/audio.wav",
        )
        assert state.input_type == "voice"

    def test_invalid_input_type(self):
        with pytest.raises(Exception):
            PharmacyState(
                input_type="text",  # not valid
                raw_input_ref="/path",
            )

    def test_full_state(self):
        state = PharmacyState(
            input_type="image",
            raw_input_ref="/bill.jpg",
            trace_id="test-trace-123",
            extracted_fields=ExtractionResult(
                items=[ExtractedBillFields(medicine_name="Aspirin")]
            ),
            validation_flags=["duplicate_batch"],
            safety_flags=[
                SafetyFlag(
                    drug_pair=("A", "B"),
                    severity="moderate",
                    description="Interaction",
                    source="test",
                )
            ],
            final_response="All clear",
            processing_time_ms=1500.0,
            estimated_cost_usd=0.005,
        )
        assert state.trace_id == "test-trace-123"
        assert len(state.extracted_fields.items) == 1
        assert len(state.safety_flags) == 1
        assert state.processing_time_ms == 1500.0

    def test_state_serialization_roundtrip(self):
        """State should serialize to dict and back."""
        state = PharmacyState(
            input_type="image",
            raw_input_ref="/test.jpg",
        )
        data = state.model_dump()
        restored = PharmacyState(**data)
        assert restored.input_type == state.input_type
        assert restored.trace_id == state.trace_id
