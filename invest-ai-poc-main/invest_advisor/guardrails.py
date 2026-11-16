
from __future__ import annotations

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from . import policy
from .observability import audit_logger, build_record, reset_turn_state


def _user_text(ctx: CallbackContext) -> str:
    content = getattr(ctx, "user_content", None)
    parts = getattr(content, "parts", None) or []
    return "\n".join(p.text for p in parts if getattr(p, "text", None))


# --- 1. INPUT GUARDRAIL (stand-in for Vertex AI Model Armor ingress) -----------------
def input_guardrail(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    reset_turn_state(state)

    raw = _user_text(callback_context)
    verdict = policy.check_input(raw)
    redaction = policy.redact_pii(raw)
    state["audit_pii_counts"] = redaction.counts
    state["audit_sanitized_query"] = redaction.text

    if verdict.blocked:
        state["audit_input_status"] = "BLOCKED_AT_INPUT"
        state["audit_input_reason"] = verdict.reason
        state["audit_output_status"] = "SKIPPED_BLOCKED_AT_INPUT"
        audit_logger.emit(build_record(callback_context, outcome="blocked_at_input"))
        return types.Content(role="model", parts=[types.Part(text=verdict.message)])

    state["audit_input_status"] = "PASSED_PII_REDACTED" if redaction.changed else "PASSED_CLEAN"
    return None


def sanitize_llm_request(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Redact PII from every user-authored text part in the request (not just the last one)."""
    for content in llm_request.contents or []:
        if content.role != "user":
            continue
        for part in content.parts or []:
            if getattr(part, "text", None):
                part.text = policy.redact_pii(part.text).text
    return None


# --- 2. OUTPUT GUARDRAIL (stand-in for Model Armor egress + FCA conduct filter) ------
def output_guardrail(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    content = llm_response.content
    if not content or not content.parts or getattr(llm_response, "partial", False):
        return None
    if any(getattr(p, "function_call", None) for p in content.parts):
        return None  # a tool call / agent transfer, not an answer for the customer
    text = "".join(
        p.text for p in content.parts if getattr(p, "text", None) and not getattr(p, "thought", False)
    )
    if not text.strip():
        return None

    state = callback_context.state
    verdict = policy.finalize_output(text, sources=list(state.get("audit_sources") or []))
    state["audit_output_status"] = verdict.status
    state["audit_output_reasons"] = verdict.reasons
    state["audit_answered_by"] = callback_context.agent_name
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=verdict.text)]))


# --- 3. OBSERVABILITY / CLOUD LOGGING -------------------------------------------------
def cloud_logging_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    state = callback_context.state
    outcome = "blocked_at_output" if str(state.get("audit_output_status", "")).startswith("BLOCKED") else "answered"
    audit_logger.emit(build_record(callback_context, outcome=outcome))
    return None
