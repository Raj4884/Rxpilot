"""
Shared pytest fixtures for RxPilot tests.

The `client` fixture is available to all test files without importing.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient with mocked database and observability dependencies."""
    with patch("api.database.check_db_health", return_value=True), \
         patch("api.database.get_connection"), \
         patch("api.database.get_recent_bills", return_value=[]), \
         patch("observability.tracing.get_langfuse", return_value=None):
        from api.main import app
        with TestClient(app) as c:
            yield c
