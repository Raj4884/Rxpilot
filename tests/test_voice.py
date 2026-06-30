"""
RxPilot — Tests for the Voice Interface (Phase 3).

Tests:
  - Transcription module (stub mode, file validation)
  - Intent parser (heuristic mode, Claude mock)
  - Answer agent (DB mock, RAG fallback, LLM stub)
  - Voice API endpoint (mocked pipeline)
  - Graph routing: voice input reaches transcribe node
  - State: VoiceQuery and VoiceAnswer Pydantic models
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import wave
import struct

import pytest

from agents.state import PharmacyState, VoiceQuery, VoiceAnswer


# ── Helpers ──────────────────────────────────


def _create_wav_file(path: str, duration_samples: int = 8000) -> None:
    """Create a minimal valid WAV file for testing."""
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        data = struct.pack('<' + 'h' * duration_samples, *([0] * duration_samples))
        wf.writeframes(data)


def _make_voice_state(audio_path: str = "/tmp/test.wav") -> PharmacyState:
    return PharmacyState(input_type="voice", raw_input_ref=audio_path)


# ══════════════════════════════════════════════
# 1. State Models
# ══════════════════════════════════════════════


class TestVoiceQueryModel:
    def test_valid_stock_query(self):
        q = VoiceQuery(
            intent="stock_query",
            drug_name="Metformin",
            question="What is the stock level of Metformin?",
        )
        assert q.intent == "stock_query"
        assert q.drug_name == "Metformin"

    def test_valid_expiry_query_no_drug(self):
        q = VoiceQuery(
            intent="expiry_query",
            drug_name=None,
            question="Which medicines expire soon?",
        )
        assert q.intent == "expiry_query"
        assert q.drug_name is None

    def test_invalid_intent_rejected(self):
        with pytest.raises(Exception):
            VoiceQuery(
                intent="bad_intent",
                drug_name=None,
                question="test",
            )

    def test_all_intents_valid(self):
        for intent in ("stock_query", "expiry_query", "interaction_query", "general_query"):
            q = VoiceQuery(intent=intent, question="test")
            assert q.intent == intent


class TestVoiceAnswerModel:
    def test_valid_answer(self):
        a = VoiceAnswer(answer_text="Metformin: 450 units.", source="database")
        assert a.confidence == 1.0
        assert a.source == "database"

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            VoiceAnswer(answer_text="test", source="database", confidence=1.5)
        with pytest.raises(Exception):
            VoiceAnswer(answer_text="test", source="database", confidence=-0.1)

    def test_valid_sources(self):
        for source in ("database", "corpus", "llm", "stub"):
            a = VoiceAnswer(answer_text="test", source=source)
            assert a.source == source


class TestPharmacyStateVoiceFields:
    def test_voice_state_defaults(self):
        s = PharmacyState(input_type="voice", raw_input_ref="/tmp/audio.wav")
        assert s.voice_query is None
        assert s.voice_answer is None
        assert s.input_type == "voice"

    def test_voice_state_with_query(self):
        q = VoiceQuery(intent="stock_query", question="How much Aspirin do we have?")
        s = PharmacyState(
            input_type="voice",
            raw_input_ref="/tmp/test.wav",
            voice_query=q,
        )
        assert s.voice_query.intent == "stock_query"

    def test_voice_state_roundtrip(self):
        q = VoiceQuery(intent="expiry_query", drug_name="Warfarin", question="When does Warfarin expire?")
        a = VoiceAnswer(answer_text="2026-12-31", source="database", confidence=0.95)
        s = PharmacyState(
            input_type="voice",
            raw_input_ref="/tmp/test.wav",
            transcript="When does Warfarin expire?",
            voice_query=q,
            voice_answer=a,
        )
        dumped = s.model_dump()
        restored = PharmacyState(**dumped)
        assert restored.voice_query.drug_name == "Warfarin"
        assert restored.voice_answer.confidence == 0.95


# ══════════════════════════════════════════════
# 2. Transcription
# ══════════════════════════════════════════════


class TestTranscription:
    def test_missing_file_raises(self):
        from voice.transcription import transcribe_audio
        with pytest.raises(FileNotFoundError):
            transcribe_audio("/nonexistent/path/audio.wav")

    def test_unsupported_format_raises(self):
        from voice.transcription import transcribe_audio
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf")
            path = f.name
        with pytest.raises(ValueError, match="Unsupported audio format"):
            transcribe_audio(path)

    def test_stub_mode_returns_string(self):
        """Without Whisper installed, should return a stub string."""
        from voice.transcription import transcribe_audio, _stub_transcription

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _create_wav_file(f.name)
            path = f.name

        # Ensure stub mode
        with patch("voice.transcription._is_whisper_available", return_value=False):
            result = transcribe_audio(path)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_stub_stock_filename(self):
        from voice.transcription import _stub_transcription
        result = _stub_transcription("stock_query.wav")
        assert "stock" in result.lower() or "metformin" in result.lower()

    def test_stub_expiry_filename(self):
        from voice.transcription import _stub_transcription
        result = _stub_transcription("expiry_check.wav")
        assert "expir" in result.lower()

    def test_stub_default_filename(self):
        from voice.transcription import _stub_transcription
        result = _stub_transcription("audio_abc123.wav")
        assert isinstance(result, str)
        assert len(result) > 5

    def test_whisper_model_called_when_available(self):
        """When Whisper IS available, should call model.transcribe."""
        from voice.transcription import transcribe_audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            _create_wav_file(f.name)
            path = f.name

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Transcribed text from Whisper"}

        with patch("voice.transcription._is_whisper_available", return_value=True), \
             patch("voice.transcription._get_model", return_value=mock_model):
            result = transcribe_audio(path)

        assert result == "Transcribed text from Whisper"
        mock_model.transcribe.assert_called_once()


# ══════════════════════════════════════════════
# 3. Intent Parser
# ══════════════════════════════════════════════


class TestIntentParser:
    """Tests for heuristic intent parsing (no Claude required)."""

    @pytest.mark.asyncio
    async def test_stock_query_heuristic(self):
        from voice.intent_parser import _heuristic_parse
        result = _heuristic_parse("How much Metformin do we have in stock?")
        assert result.intent == "stock_query"

    @pytest.mark.asyncio
    async def test_expiry_query_heuristic(self):
        from voice.intent_parser import _heuristic_parse
        result = _heuristic_parse("Which medicines are expiring in the next 30 days?")
        assert result.intent == "expiry_query"

    @pytest.mark.asyncio
    async def test_interaction_query_heuristic(self):
        from voice.intent_parser import _heuristic_parse
        result = _heuristic_parse("Are there any drug interactions between Warfarin and Aspirin?")
        assert result.intent == "interaction_query"

    @pytest.mark.asyncio
    async def test_general_query_heuristic(self):
        from voice.intent_parser import _heuristic_parse
        result = _heuristic_parse("Tell me about the last delivery.")
        assert result.intent == "general_query"

    @pytest.mark.asyncio
    async def test_drug_name_extracted(self):
        from voice.intent_parser import _heuristic_parse
        result = _heuristic_parse("What is the stock level of Metformin?")
        assert result.drug_name is not None
        assert "Metformin" in result.drug_name or "metformin" in result.drug_name.lower()

    @pytest.mark.asyncio
    async def test_empty_transcript_returns_general(self):
        from voice.intent_parser import parse_intent
        result = await parse_intent("")
        assert result.intent == "general_query"
        assert "(empty" in result.question.lower()

    @pytest.mark.asyncio
    async def test_parse_intent_no_api_key_uses_heuristics(self):
        """Without API key, should fall back to heuristics."""
        from voice.intent_parser import parse_intent
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            result = await parse_intent("What stock do we have of Aspirin?")
        assert result.intent == "stock_query"

    @pytest.mark.asyncio
    async def test_claude_parse_success(self):
        """With a valid Claude response, should return parsed intent."""
        from voice.intent_parser import parse_intent

        mock_response = MagicMock()
        mock_response.content = [MagicMock(
            text='{"intent": "expiry_query", "drug_name": "Warfarin", "question": "When does Warfarin expire?"}'
        )]

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
             patch("voice.intent_parser.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response
            result = await parse_intent("When does Warfarin expire?")

        assert result.intent == "expiry_query"
        assert result.drug_name == "Warfarin"

    @pytest.mark.asyncio
    async def test_claude_parse_bad_json_falls_back(self):
        """If Claude returns bad JSON, should fall back to heuristics."""
        from voice.intent_parser import parse_intent

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="NOT JSON")]

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
             patch("voice.intent_parser.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response
            result = await parse_intent("How much stock do we have?")

        # Should fall back to heuristics and still return valid result
        assert result.intent in ("stock_query", "expiry_query", "interaction_query", "general_query")


# ══════════════════════════════════════════════
# 4. Answer Agent
# ══════════════════════════════════════════════


class TestAnswerAgent:
    @pytest.mark.asyncio
    async def test_stock_query_no_db(self):
        """Stock query with no DB should return graceful stub answer."""
        from voice.answer_agent import answer_query
        q = VoiceQuery(intent="stock_query", drug_name="Metformin", question="Stock of Metformin?")
        with patch("voice.answer_agent._get_connection", return_value=None):
            result = await answer_query(q)
        assert isinstance(result.answer_text, str)
        assert len(result.answer_text) > 5
        assert result.source == "stub"

    @pytest.mark.asyncio
    async def test_stock_query_with_db(self):
        """Stock query should return DB result when available."""
        from voice.answer_agent import answer_query, _answer_stock_query
        import psycopg2.extras

        mock_conn = MagicMock()
        mock_cur = MagicMock(spec=["fetchall", "execute"])
        mock_cur.fetchall.return_value = [
            {
                "medicine_name": "Metformin 500mg",
                "total_qty": 450,
                "bill_count": 3,
                "last_received": MagicMock(strftime=lambda f: "15 Nov 2024"),
            }
        ]
        mock_conn.cursor.return_value = mock_cur

        result = _answer_stock_query("Metformin", mock_conn)

        assert "Metformin" in result.answer_text
        assert "450" in result.answer_text
        assert result.source == "database"
        assert result.confidence > 0.9

    @pytest.mark.asyncio
    async def test_expiry_query_no_db(self):
        """Expiry query with no DB returns graceful stub."""
        from voice.answer_agent import answer_query
        q = VoiceQuery(intent="expiry_query", drug_name=None, question="What expires soon?")
        with patch("voice.answer_agent._get_connection", return_value=None):
            result = await answer_query(q)
        assert result.source == "stub"

    @pytest.mark.asyncio
    async def test_expiry_query_no_results(self):
        """Expiry query with no matching rows should return informative message."""
        from voice.answer_agent import _answer_expiry_query

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cur

        result = _answer_expiry_query(None, mock_conn)
        assert "no medicines" in result.answer_text.lower() or "not found" in result.answer_text.lower()
        assert result.source == "database"

    @pytest.mark.asyncio
    async def test_interaction_query_with_known_drug(self):
        """Interaction query for a drug in the CSV corpus."""
        from voice.answer_agent import answer_query
        q = VoiceQuery(intent="interaction_query", drug_name="Warfarin", question="Warfarin interactions?")
        result = await answer_query(q)
        # Should find something in the CSV corpus
        assert result.source in ("corpus", "stub")
        assert isinstance(result.answer_text, str)

    @pytest.mark.asyncio
    async def test_interaction_query_no_drug_name(self):
        """Interaction query without drug name should ask for specification."""
        from voice.answer_agent import _answer_interaction_query
        result = _answer_interaction_query(None, "Are there interactions?")
        assert "specify" in result.answer_text.lower() or "drug name" in result.answer_text.lower()

    @pytest.mark.asyncio
    async def test_general_query_no_api_key(self):
        """General query without API key returns informative stub."""
        from voice.answer_agent import _answer_general_query
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            result = await _answer_general_query("Tell me something useful.")
        assert result.source == "stub"
        assert len(result.answer_text) > 10

    @pytest.mark.asyncio
    async def test_general_query_with_claude(self):
        """General query with Claude should return LLM answer."""
        from voice.answer_agent import _answer_general_query

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is a test pharmacy answer. Please verify with a licensed pharmacist.")]

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
             patch("voice.answer_agent.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response
            result = await _answer_general_query("What is Metformin used for?")

        assert result.source == "llm"
        assert "pharmacy" in result.answer_text.lower()


# ══════════════════════════════════════════════
# 5. Voice API Endpoint
# ══════════════════════════════════════════════


class TestVoiceEndpoint:
    """Tests for POST /v1/voice."""

    def _make_wav_bytes(self) -> bytes:
        """Create minimal valid WAV bytes for upload testing."""
        import io
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(struct.pack('<' + 'h' * 800, *([0] * 800)))
        return buf.getvalue()

    def test_voice_no_file_returns_422(self, client):
        res = client.post("/v1/voice")
        assert res.status_code == 422

    def test_voice_wrong_format_returns_422(self, client):
        res = client.post(
            "/v1/voice",
            files={"file": ("document.pdf", b"%PDF-fake", "application/pdf")},
        )
        assert res.status_code == 422

    def test_voice_wav_file_returns_200(self, client):
        """WAV upload should return 200 with voice response fields."""
        mock_graph_result = {
            "input_type": "voice",
            "raw_input_ref": "/uploads/voice/test.wav",
            "transcript": "What is the stock level of Metformin?",
            "voice_query": {
                "intent": "stock_query",
                "drug_name": "Metformin",
                "question": "What is the stock level of Metformin?",
            },
            "voice_answer": {
                "answer_text": "Metformin 500mg: 450 units across 3 bills. Last received: 15 Nov 2024.",
                "source": "database",
                "confidence": 0.95,
            },
            "validation_flags": [],
            "safety_flags": [],
            "processing_time_ms": 1234.0,
            "estimated_cost_usd": 0.0,
            "error": None,
            "extracted_fields": None,
            "forecast": None,
            "final_response": None,
            "voice_query": {
                "intent": "stock_query",
                "drug_name": "Metformin",
                "question": "What is the stock level of Metformin?",
            },
            "voice_answer": {
                "answer_text": "Metformin 500mg: 450 units.",
                "source": "database",
                "confidence": 0.95,
            },
        }

        with patch("api.routes.voice.compiled_graph") as mock_graph, \
             patch("api.routes.voice.flush_langfuse"):
            mock_graph.ainvoke = AsyncMock(return_value=mock_graph_result)

            wav_bytes = self._make_wav_bytes()
            res = client.post(
                "/v1/voice",
                files={"file": ("query.wav", wav_bytes, "audio/wav")},
            )

        assert res.status_code == 200
        data = res.json()
        assert "transcript" in data
        assert "intent" in data
        assert "answer" in data
        assert "answer_source" in data
        assert "answer_confidence" in data
        assert "processing_time_ms" in data
        assert "trace_id" in data

    def test_voice_response_fields(self, client):
        """Check specific response field values."""
        mock_graph_result = {
            "input_type": "voice",
            "raw_input_ref": "/uploads/voice/test.wav",
            "transcript": "Which medicines expire soon?",
            "voice_query": {
                "intent": "expiry_query",
                "drug_name": None,
                "question": "Which medicines expire soon?",
            },
            "voice_answer": {
                "answer_text": "3 medicines expiring within 60 days: Aspirin (2025-01-15); Warfarin (2025-02-01); Metformin (2025-03-10).",
                "source": "database",
                "confidence": 0.95,
            },
            "validation_flags": [],
            "safety_flags": [],
            "processing_time_ms": 800.0,
            "estimated_cost_usd": 0.0,
            "error": None,
            "extracted_fields": None,
            "forecast": None,
            "final_response": None,
        }

        with patch("api.routes.voice.compiled_graph") as mock_graph, \
             patch("api.routes.voice.flush_langfuse"):
            mock_graph.ainvoke = AsyncMock(return_value=mock_graph_result)

            wav_bytes = self._make_wav_bytes()
            res = client.post(
                "/v1/voice",
                files={"file": ("expiry.wav", wav_bytes, "audio/wav")},
            )

        assert res.status_code == 200
        data = res.json()
        assert data["intent"] == "expiry_query"
        assert data["drug_name"] is None
        assert "expiring" in data["answer"].lower() or "medicines" in data["answer"].lower()
        assert data["answer_source"] == "database"
        assert data["answer_confidence"] == 0.95


# ══════════════════════════════════════════════
# 6. Graph Routing
# ══════════════════════════════════════════════


class TestGraphRouting:
    def test_voice_input_routes_to_transcribe(self):
        """route_by_input_type should return 'transcribe' for voice input."""
        from graph import route_by_input_type
        state = {"input_type": "voice", "raw_input_ref": "/tmp/audio.wav"}
        assert route_by_input_type(state) == "transcribe"

    def test_image_input_routes_to_extract(self):
        """route_by_input_type should return 'extract' for image input."""
        from graph import route_by_input_type
        state = {"input_type": "image", "raw_input_ref": "/tmp/bill.jpg"}
        assert route_by_input_type(state) == "extract"

    def test_missing_input_type_defaults_to_extract(self):
        """Missing input_type should default to extract (image) path."""
        from graph import route_by_input_type
        state = {"raw_input_ref": "/tmp/bill.jpg"}
        assert route_by_input_type(state) == "extract"
