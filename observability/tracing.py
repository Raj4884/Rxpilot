"""
RxPilot — Langfuse tracing wrapper.

Every agent call is wrapped in a Langfuse trace/span that records:
  - Input (image ref, prompt, etc.)
  - Output (extracted fields, flags, etc.)
  - Token usage
  - Latency (ms)
  - Estimated cost (USD)

Usage:
    from observability.tracing import get_langfuse, create_trace, traced_generation

    trace = create_trace(trace_id="...", name="bill-extraction")
    span = trace.span(name="extraction-agent", input={"image": "..."})
    # ... do work ...
    span.end(output={"items": [...]}, metadata={"tokens": 1500})
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

from langfuse import Langfuse

logger = logging.getLogger(__name__)

# ── Singleton Langfuse client ──
_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse:
    """Get or initialize the Langfuse client from environment variables."""
    global _langfuse
    if _langfuse is None:
        try:
            _langfuse = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-rxpilot-dev"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-rxpilot-dev"),
                host=os.getenv("LANGFUSE_HOST", "http://localhost:3001"),
            )
            logger.info(
                "Langfuse client initialized — host=%s",
                os.getenv("LANGFUSE_HOST", "http://localhost:3001"),
            )
        except Exception as e:
            logger.warning("Langfuse initialization failed: %s — tracing disabled", e)
            _langfuse = None  # type: ignore[assignment]
    return _langfuse  # type: ignore[return-value]


def create_trace(
    trace_id: str,
    name: str,
    input_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """
    Create a new Langfuse trace for a pipeline run.

    Returns a trace object (or a no-op stub if Langfuse is unavailable).
    """
    lf = get_langfuse()
    if lf is None:
        return _NoOpTrace()

    try:
        trace = lf.trace(
            id=trace_id,
            name=name,
            input=input_data or {},
            metadata=metadata or {},
        )
        logger.debug("Created Langfuse trace: %s (%s)", trace_id, name)
        return trace
    except Exception as e:
        logger.warning("Failed to create Langfuse trace: %s", e)
        return _NoOpTrace()


@contextmanager
def traced_span(
    trace: Any,
    name: str,
    input_data: dict[str, Any] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager that creates a Langfuse span, tracks timing,
    and finalizes on exit.

    Usage:
        with traced_span(trace, "extraction", input_data={"image": path}) as ctx:
            result = do_extraction(...)
            ctx["output"] = result
            ctx["usage"] = {"input_tokens": 500, "output_tokens": 200}
    """
    ctx: dict[str, Any] = {
        "output": None,
        "usage": None,
        "metadata": {},
        "level": "DEFAULT",
        "status_message": None,
    }
    start_time = time.perf_counter()

    try:
        yield ctx
    except Exception as e:
        ctx["level"] = "ERROR"
        ctx["status_message"] = str(e)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        ctx["metadata"]["latency_ms"] = round(elapsed_ms, 2)

        try:
            if hasattr(trace, "span"):
                span = trace.span(
                    name=name,
                    input=input_data or {},
                    output=ctx.get("output") or {},
                    metadata=ctx.get("metadata", {}),
                    level=ctx.get("level", "DEFAULT"),
                    status_message=ctx.get("status_message"),
                )
                # Record token usage if available
                if ctx.get("usage"):
                    span.update(
                        usage=ctx["usage"],
                    )
        except Exception as e:
            logger.warning("Failed to record Langfuse span '%s': %s", name, e)


def traced_generation(
    trace: Any,
    name: str,
    model: str,
    input_data: dict[str, Any] | None = None,
) -> Any:
    """
    Create a Langfuse generation span for an LLM call.

    Returns a generation object that should be updated with output/usage on completion.
    """
    try:
        if hasattr(trace, "generation"):
            return trace.generation(
                name=name,
                model=model,
                input=input_data or {},
            )
    except Exception as e:
        logger.warning("Failed to create Langfuse generation '%s': %s", name, e)

    return _NoOpGeneration()


def flush_langfuse() -> None:
    """Flush any pending Langfuse events."""
    lf = get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception as e:
            logger.warning("Langfuse flush failed: %s", e)


# ── No-op stubs for when Langfuse is unavailable ──


class _NoOpTrace:
    """Stub trace that silently ignores all calls."""

    def span(self, **kwargs: Any) -> "_NoOpTrace":
        return self

    def generation(self, **kwargs: Any) -> "_NoOpGeneration":
        return _NoOpGeneration()

    def update(self, **kwargs: Any) -> None:
        pass

    def end(self, **kwargs: Any) -> None:
        pass


class _NoOpGeneration:
    """Stub generation that silently ignores all calls."""

    def update(self, **kwargs: Any) -> None:
        pass

    def end(self, **kwargs: Any) -> None:
        pass
