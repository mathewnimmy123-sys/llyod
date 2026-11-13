"""Audit / observability sink.

Three sinks, all best-effort (logging must NEVER break the customer response):
  1. stdout JSON line  - on Cloud Run / GKE this is ingested by Cloud Logging automatically
                         (a `severity` field maps to the log level).
  2. local JSONL file  - handy for `adk web` demos and tests (LOCAL_AUDIT_LOG=true).
  3. Cloud Logging API - direct `log_struct` when ENABLE_CLOUD_LOGGING=true. Requires GCP
                         credentials (ADC / service account); an AI Studio API key is NOT enough.

Records hold redacted text only, PII *counts* (never values) and chunk ids - so they are
safe to retain for audit and to feed dashboards / log-based metrics (block rate, latency).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

from . import config

_stdout = logging.getLogger("invest_ai.audit")
if not _stdout.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _stdout.addHandler(_h)
    _stdout.setLevel(logging.INFO)
    _stdout.propagate = False

AUDIT_KEYS_DEFAULTS: dict[str, Any] = {
    "audit_input_status": "N/A",
    "audit_input_reason": "",
    "audit_pii_counts": {},
    "audit_sanitized_query": "",
    "audit_answered_by": "",
    "audit_tools": [],
    "audit_sources": [],
    "audit_output_status": "N/A",
    "audit_output_reasons": [],
}


def reset_turn_state(state: Any) -> None:
    """Called at the start of every turn so values from a previous turn never leak."""
    state["audit_started_at"] = time.time()
    for key, default in AUDIT_KEYS_DEFAULTS.items():
        state[key] = type(default)() if isinstance(default, (list, dict)) else default


def build_record(ctx: Any, *, outcome: str) -> dict[str, Any]:
    state = ctx.state
    started = state.get("audit_started_at") or time.time()
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": "WARNING" if outcome.startswith("blocked") else "INFO",
        "event": "invest_ai_turn",
        "outcome": outcome,
        "pipeline": getattr(ctx, "agent_name", ""),
        "invocation_id": getattr(ctx, "invocation_id", ""),
        "session_id": getattr(getattr(ctx, "session", None), "id", ""),
        "latency_ms": int((time.time() - started) * 1000),
    }
    for key in AUDIT_KEYS_DEFAULTS:
        record[key.removeprefix("audit_")] = state.get(key, AUDIT_KEYS_DEFAULTS[key])
    record["sanitized_query"] = str(record["sanitized_query"])[:500]
    return record


class AuditLogger:
    def __init__(self) -> None:
        self._cloud = None
        self._cloud_failed = False

    def _cloud_logger(self):
        if not config.ENABLE_CLOUD_LOGGING or self._cloud_failed:
            return None
        if self._cloud is None:
            try:
                import google.cloud.logging as gcl  # optional dependency

                client = gcl.Client(project=config.GCP_PROJECT or None)
                self._cloud = client.logger(config.CLOUD_LOG_NAME)
            except Exception as exc:  # noqa: BLE001 - never raise from logging
                self._cloud_failed = True
                _stdout.warning(json.dumps({"severity": "WARNING", "message": f"cloud logging disabled: {exc}"}))
        return self._cloud

    def emit(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        try:
            _stdout.info(line)
        except Exception:  # noqa: BLE001
            pass
        if config.LOCAL_AUDIT_LOG:
            try:
                config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(config.LOG_FILE, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass  # read-only filesystem on some runtimes; stdout sink still has it
        cloud = self._cloud_logger()
        if cloud is not None:
            try:
                cloud.log_struct(record, severity=record.get("severity", "INFO"))
            except Exception:  # noqa: BLE001
                pass


audit_logger = AuditLogger()
