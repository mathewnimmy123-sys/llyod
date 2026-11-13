"""Central configuration. Everything is overridable by environment variable / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# `adk web` loads <agent_dir>/.env itself; we also read it (and a repo-root .env) so the
# code behaves the same under pytest, scripts and Cloud Run. Real env vars win.
load_dotenv(PACKAGE_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

# The AI Studio key can be supplied as GEMINI_API_KEY (old demo) or GOOGLE_API_KEY (ADK).
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
# AI Studio key => not Vertex. Set GOOGLE_GENAI_USE_VERTEXAI=TRUE (+ project/location) to
# switch to Vertex AI with no code change.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


MODEL = os.getenv("INVEST_AI_MODEL", "gemini-3.8-flash")
DATA_FILE = PROJECT_ROOT / "data" / "uk_investment_guidance_2026.txt"

# Audit logging sinks
LOCAL_AUDIT_LOG = _flag("LOCAL_AUDIT_LOG", True)
LOG_FILE = PROJECT_ROOT / "logs" / "cloud_audit_logs.jsonl"
ENABLE_CLOUD_LOGGING = _flag("ENABLE_CLOUD_LOGGING", False)  # needs GCP credentials (ADC)
CLOUD_LOG_NAME = os.getenv("CLOUD_LOG_NAME", "invest-ai-audit")
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
