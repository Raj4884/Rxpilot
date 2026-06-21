"""
Tests for agents.extraction_agent — extraction with mocked Claude API.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.extraction_agent import (
    _parse_extraction_response,
    _estimate_cost,
    run_extraction,
)
from agents.state import PharmacyState


class TestParseExtractionResponse:
    """Tests for the JSON response parser."""

    def test_clean_json_object(self):
        raw = json.dumps({
            "items": [
                {"medicine_name": "Paracetamol", "quantity": 10}
            ]
        })
        result = _parse_extraction_response(raw)
        assert len(result) == 1
        assert result[0]["medicine_name"] == "Paracetamol"

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"items": [{"medicine_name": "Aspirin"}]}\n```'
        result = _parse_extraction_response(raw)
        assert len(result) == 1
        assert result[0]["medicine_name"] == "Aspirin"

    def test_bare_array(self):
        raw = json.dumps([
            {"medicine_name": "Med A"},
            {"medicine_name": "Med B"},
        ])
        result = _parse_extraction_response(raw)
        assert len(result) == 2

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"items": [{"medicine_name": "Test"}]}\nDone.'
        result = _parse_extraction_response(raw)
        assert len(result) == 1

    def test_empty_items(self):
        raw = json.dumps({"items": []})
        result = _parse_extraction_response(raw)
        assert result == []

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            _parse_extraction_response("not json at all")


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_zero_tokens(self):
        assert _estimate_cost(0, 0) == 0.0

    def test_known_values(self):
        # 1M input tokens = $3, 1M output tokens = $15
        cost = _estimate_cost(1_000_000, 1_000_000)
        assert cost == pytest.approx(18.0, abs=0.01)

    def test_small_call(self):
        # 1000 input + 500 output
        cost = _estimate_cost(1000, 500)
        assert cost > 0
        assert cost < 0.02  # ~$0.0105 for 1K input + 500 output


class TestRunExtraction:
    """Tests for the extraction agent with mocked Claude API."""

    @pytest.fixture
    def temp_image(self, tmp_path: Path) -> str:
        """Create a minimal test image file."""
        img_path = tmp_path / "test_bill.jpg"
        # Write a minimal JPEG-like file (just needs to exist for base64 encoding)
        img_path.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 100)
        return str(img_path)

    @pytest.fixture
    def mock_claude_response(self):
        """Create a mock Claude API response."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps({
            "items": [
                {
                    "medicine_name": "Paracetamol 500mg",
                    "batch_number": "B2024-001",
                    "expiry_date": "2026-06-01",
                    "quantity": 100,
                    "unit": "tablets",
                    "price": 25.50,
                    "currency": "INR",
                }
            ]
        })
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 1500
        mock_response.usage.output_tokens = 200
        return mock_response

    @pytest.mark.asyncio
    async def test_successful_extraction(self, temp_image, mock_claude_response):
        """Test a successful extraction with mocked Claude."""
        state = PharmacyState(
            input_type="image",
            raw_input_ref=temp_image,
        )

        with patch("agents.extraction_agent._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_claude_response

            result = await run_extraction(state)

        assert result.extracted_fields is not None
        assert len(result.extracted_fields.items) == 1
        assert result.extracted_fields.items[0].medicine_name == "Paracetamol 500mg"
        assert result.extracted_fields.items[0].price == 25.50
        assert result.processing_time_ms > 0
        assert result.estimated_cost_usd > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_extraction_with_parse_retry(self, temp_image, mock_claude_response):
        """Test that extraction retries on parse failure."""
        state = PharmacyState(
            input_type="image",
            raw_input_ref=temp_image,
        )

        # First response is invalid, second is valid
        bad_response = MagicMock()
        bad_response.content = [MagicMock()]
        bad_response.content[0].text = "not valid json"
        bad_response.usage = MagicMock()
        bad_response.usage.input_tokens = 1000
        bad_response.usage.output_tokens = 50

        with patch("agents.extraction_agent._get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = [
                bad_response,
                mock_claude_response,
            ]

            result = await run_extraction(state)

        assert result.extracted_fields is not None
        assert len(result.extracted_fields.items) == 1
        assert result.extracted_fields.parse_retries == 1

    @pytest.mark.asyncio
    async def test_extraction_all_retries_exhausted(self, temp_image):
        """Test that extraction fails gracefully after all retries."""
        state = PharmacyState(
            input_type="image",
            raw_input_ref=temp_image,
        )

        bad_response = MagicMock()
        bad_response.content = [MagicMock()]
        bad_response.content[0].text = "not json"
        bad_response.usage = MagicMock()
        bad_response.usage.input_tokens = 500
        bad_response.usage.output_tokens = 20

        with patch("agents.extraction_agent._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = bad_response

            result = await run_extraction(state)

        assert result.extracted_fields is not None
        assert len(result.extracted_fields.items) == 0
        assert result.error is not None
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_image_raises(self):
        """Test that a missing image file causes an error."""
        state = PharmacyState(
            input_type="image",
            raw_input_ref="/nonexistent/bill.jpg",
        )

        with patch("agents.extraction_agent._get_client"):
            # FileNotFoundError happens before the retry loop (in _encode_image),
            # so it propagates through the retry loop and sets state.error
            result = await run_extraction(state)

        # The error should be captured in the state after retries exhaust
        assert result.error is not None
        assert "not found" in result.error.lower() or "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_multi_item_extraction(self, temp_image):
        """Test extraction of multiple items from a bill."""
        state = PharmacyState(
            input_type="image",
            raw_input_ref=temp_image,
        )

        multi_response = MagicMock()
        multi_response.content = [MagicMock()]
        multi_response.content[0].text = json.dumps({
            "items": [
                {"medicine_name": "Paracetamol 500mg", "price": 25.0},
                {"medicine_name": "Amoxicillin 250mg", "price": 45.0},
                {"medicine_name": "Omeprazole 20mg", "price": 30.0},
            ]
        })
        multi_response.usage = MagicMock()
        multi_response.usage.input_tokens = 2000
        multi_response.usage.output_tokens = 400

        with patch("agents.extraction_agent._get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = multi_response

            result = await run_extraction(state)

        assert result.extracted_fields is not None
        assert len(result.extracted_fields.items) == 3
        names = [i.medicine_name for i in result.extracted_fields.items]
        assert "Paracetamol 500mg" in names
        assert "Amoxicillin 250mg" in names
        assert "Omeprazole 20mg" in names
