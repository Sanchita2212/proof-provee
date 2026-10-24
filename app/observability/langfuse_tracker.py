import os
import json
import time
import logging
from typing import Dict, Any, Optional
from app.config import settings

logger = logging.getLogger(__name__)

class ObservabilityManager:
    """
    Manages tracing, latency, and cost instrumentation.
    Integrates with Langfuse when configured; gracefully degrades to local structured tracing
    if Langfuse is disabled, unconfigured, or unreachable.
    Captures the required four trace steps (query_compilation, source_discrimination,
    evidence_coverage, answer_generation), alongside latency and model usage/cost.
    """
    def __init__(self):
        self.langfuse_client = None
        if settings.LANGFUSE_ENABLED and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            try:
                from langfuse import Langfuse
                self.langfuse_client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                logger.info("Langfuse observability initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize Langfuse: {e}. Falling back to local tracing.")

    def start_trace(self, name: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        if self.langfuse_client:
            try:
                trace_id = self.langfuse_client.create_trace_id()
                span = self.langfuse_client.start_observation(
                    name=name,
                    as_type="span",
                    metadata=metadata or {},
                )
                return LangfuseTraceWrapper(span=span, client=self.langfuse_client)
            except Exception as e:
                logger.warning(f"Failed to start Langfuse trace: {e}")
        return LocalTrace(name=name, metadata=metadata)

    def flush(self):
        if self.langfuse_client:
            try:
                self.langfuse_client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse client: {e}")

class LangfuseTraceWrapper:
    def __init__(self, span: Any, client: Any):
        self.root_span = span
        self.client = client
        self.start_time = time.time()

    def log_step(
        self,
        step_name: str,
        description: str,
        input_data: Any,
        output_data: Any,
        latency_ms: Optional[float] = None,
        model: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        cost: Optional[float] = None,
    ):
        try:
            span_meta = {
                "step_name": step_name,
                "description": description,
                "latency_ms": latency_ms,
            }
            if usage:
                span_meta["usage"] = usage
            if cost is not None:
                span_meta["cost"] = cost

            sub_span = self.client.start_observation(
                name=step_name,
                as_type="generation" if step_name == "answer_generation" else "span",
                input=str(input_data),
                output=str(output_data),
                metadata=span_meta,
                model=model,
            )
            sub_span.end()
        except Exception as e:
            logger.debug(f"Error logging Langfuse sub-span {step_name}: {e}")

    def update(self, output: Any = None, metadata: Optional[Dict[str, Any]] = None):
        try:
            self.root_span.update(output=output, metadata=metadata)
            self.root_span.end()
        except Exception as e:
            logger.debug(f"Error updating Langfuse root span: {e}")

class LocalTrace:
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {}
        self.start_time = time.time()
        self.steps = []

    def log_step(
        self,
        step_name: str,
        description: str,
        input_data: Any,
        output_data: Any,
        latency_ms: Optional[float] = None,
        model: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        cost: Optional[float] = None,
    ):
        self.steps.append({
            "step_name": step_name,
            "description": description,
            "input": str(input_data),
            "output": str(output_data),
            "latency_ms": latency_ms,
            "model": model,
            "usage": usage,
            "cost": cost,
        })

    def update(self, output: Any = None, metadata: Optional[Dict[str, Any]] = None):
        if metadata:
            self.metadata.update(metadata)

observability = ObservabilityManager()
