"""
Tests for API endpoints — health and upload routes.

Uses FastAPI's TestClient with mocked dependencies for isolated testing.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient with mocked database."""
    with patch("api.database.check_db_health", return_value=True), \
         patch("api.database.get_connection"), \
         patch("api.database.get_recent_bills", return_value=[]), \
         patch("observability.tracing.get_langfuse", return_value=None):
        from api.main import app
        with TestClient(app) as c:
            yield c


class TestRootEndpoint:
    """Tests for GET /"""

    def test_root_returns_api_info(self, client):
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "RxPilot"
        assert "version" in data
        assert "disclaimer" in data


class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_healthy(self, client):
        with patch("api.routes.health.check_db_health", return_value=True), \
             patch("api.routes.health.os.getenv", side_effect=lambda k, d="": "fake-key" if k == "ANTHROPIC_API_KEY" else d):
            res = client.get("/health")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "healthy"

    def test_health_degraded_no_db(self, client):
        with patch("api.routes.health.check_db_health", return_value=False):
            res = client.get("/health")
            assert res.status_code == 200
            data = res.json()
            assert data["services"]["database"] == "unavailable"


class TestBillsEndpoint:
    """Tests for GET /v1/bills"""

    def test_list_bills_empty(self, client):
        with patch("api.routes.upload.get_recent_bills", return_value=[]):
            res = client.get("/v1/bills")
            assert res.status_code == 200
            data = res.json()
            assert data["bills"] == []

    def test_list_bills_with_data(self, client):
        mock_bills = [
            {
                "id": "123",
                "image_path": "/uploads/test.jpg",
                "status": "completed",
                "trace_id": "trace-1",
                "processing_time_ms": 1500.0,
                "estimated_cost_usd": 0.005,
                "extracted_data": {"items": [{"medicine_name": "Test"}]},
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        with patch("api.routes.upload.get_recent_bills", return_value=mock_bills):
            res = client.get("/v1/bills")
            assert res.status_code == 200
            data = res.json()
            assert len(data["bills"]) == 1
            assert data["bills"][0]["items_count"] == 1


class TestUploadEndpoint:
    """Tests for POST /v1/upload"""

    def test_upload_no_file(self, client):
        """Upload without a file should fail."""
        res = client.post("/v1/upload")
        assert res.status_code == 422

    def test_upload_wrong_file_type(self, client):
        """Upload with a non-image file should fail."""
        res = client.post(
            "/v1/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert res.status_code == 400
        assert "Unsupported file type" in res.json()["detail"]

    def test_upload_valid_image(self, client):
        """Upload a valid image and run the pipeline (mocked)."""
        mock_result = {
            "input_type": "image",
            "raw_input_ref": "/uploads/test.jpg",
            "trace_id": "test-trace",
            "extracted_fields": {
                "items": [
                    {
                        "medicine_name": "Paracetamol",
                        "batch_number": "B001",
                        "expiry_date": "2026-01-01",
                        "manufacture_date": None,
                        "quantity": 50,
                        "unit": "tablets",
                        "supplier_name": None,
                        "price": 25.0,
                        "currency": "INR",
                    }
                ],
                "raw_llm_output": "{}",
                "parse_retries": 0,
            },
            "processing_time_ms": 1234.5,
            "estimated_cost_usd": 0.003,
            "error": None,
            "validation_flags": [],
            "safety_flags": [],
            "forecast": None,
            "final_response": None,
            "transcript": None,
        }

        with patch("api.routes.upload.compiled_graph") as mock_graph, \
             patch("api.routes.upload.insert_bill", return_value="bill-123"), \
             patch("api.routes.upload.flush_langfuse"):
            mock_graph.ainvoke = AsyncMock(return_value=mock_result)

            # Create a minimal JPEG-like file
            fake_image = b'\xff\xd8\xff\xe0' + b'\x00' * 100
            res = client.post(
                "/v1/upload",
                files={"file": ("bill.jpg", fake_image, "image/jpeg")},
            )

            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "completed"
            assert data["items_count"] == 1
            assert data["items"][0]["medicine_name"] == "Paracetamol"
