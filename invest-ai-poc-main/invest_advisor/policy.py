"""Pure-Python guardrail rules (no ADK imports, so they are fully unit-testable).

`guardrails.py` wires these functions into ADK callbacks. Keeping the rules here means
CI can test them in milliseconds without a model or network, and they can later be
swapped for / layered under Vertex AI Model Armor without touching the agent wiring.

NOTE: regex deny-lists are a *baseline*, not a complete defence. In production, layer
Model Armor (prompt-injection, PII/DLP, harmful content) in front of these rules.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 2000

# --------------------------------------------------------------------------- messages
INPUT_BLOCK_MESSAGE = (
    "[Input Guardrail Blocked]: Your request could not be processed because it appears to "
    "conflict with our security policy. Please rephrase your investment question."
)
INPUT_TOO_LONG_MESSAGE = (
    "[Input Guardrail Blocked]: Your message is too long. Please shorten it and try again."
)
OUTPUT_BLOCK_MESSAGE = (
    "[Output Guardrail Blocked]: I can explain how UK pensions and ISAs work, but I can't "
    "give a personal recommendation. For advice tailored to you, please speak to a "
    "regulated financial adviser."
)
DISCLAIMER = (
    "\n\n---\n*Regulatory Notice: This information is for guidance only and does not "
    "constitute regulated personal financial advice. Capital is at risk; tax rules depend "
    "on individual circumstances.*"
)

# --------------------------------------------------------------------- input: injection
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def normalise(text: str) -> str:
    """NFKC-fold, strip zero-width chars and collapse whitespace so trivial obfuscation
    (e.g. 'ig\u200bnore   previous') does not defeat the patterns."""
    folded = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", folded).strip().lower()


INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern)
    for name, pattern in {
        "ignore_instructions": (
            r"\b(ignore|disregard|forget|override)\b.{0,30}"
            r"\b(previous|prior|above|earlier|all|your|system)\b.{0,30}"
            r"\b(instructions?|rules?|guidelines?|prompts?)\b"
        ),
        "reveal_prompt": (
            r"\b(reveal|show|print|repeat|leak|output|display)\b.{0,30}"
            r"\b(system|hidden|initial|developer)\b.{0,15}\b(prompt|instructions?|message)\b"
        ),
        "bypass_compliance": (
            r"\b(bypass|disable|turn off|skip|circumvent)\b.{0,30}"
            r"\b(compliance|guardrails?|safety|filters?|fca)\b"
        ),
        "jailbreak": r"\b(jailbreak|dan mode|developer mode|do anything now)\b",
        "role_override": (
            r"\b(you are now|pretend (to be|you are)|act as)\b.{0,40}"
            r"\b(unrestricted|no restrictions|unfiltered|without (any )?(rules|restrictions|limits))\b"
        ),
    }.items()
}


@dataclass(frozen=True)
class InputVerdict:
    blocked: bool
    reason: str = ""  # machine-readable, safe to log (never the raw text)
    message: str = ""  # what the user sees when blocked


def check_input(text: str) -> InputVerdict:
    if len(text) > MAX_INPUT_CHARS:
        return InputVerdict(True, "input_too_long", INPUT_TOO_LONG_MESSAGE)
    norm = normalise(text)
    for name, pattern in INJECTION_PATTERNS.items():
        if pattern.search(norm):
            return InputVerdict(True, f"injection:{name}", INPUT_BLOCK_MESSAGE)
    return InputVerdict(False)


# ----------------------------------------------------------------------- input: PII
def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _redact_cards(text: str) -> tuple[str, int]:
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            count += 1
            return "[REDACTED_CARD]"
        return m.group(0)

    return re.sub(r"(?<![\w£.,])(?:\d[ -]?){13,19}(?!\d)", repl, text), count


# Order matters: more specific / longer patterns first.
_PII_REGEXES: list[tuple[str, re.Pattern[str]]] = [
    ("IBAN", re.compile(r"\bGB\d{2}\s?[A-Z]{4}(?:\s?\d){14}\b", re.I)),
    ("NINO", re.compile(r"\b[A-Za-z]{2}(?:\s?\d{2}){3}\s?[A-Da-d]\b")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("SORT_CODE", re.compile(r"(?<![\w£.,])\d{2}-\d{2}-\d{2}(?!\d)")),
    ("PHONE", re.compile(r"(?<![\w£.,])(?:\+44\s?\(?0?\)?\s?|0)(?:\d\s?){9,10}(?!\d)")),
    ("POSTCODE", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.I)),
    ("ACCOUNT_NO", re.compile(r"(?i)\b(account(?:\s+(?:no|number|num))?\.?:?\s*)\d{8}\b")),
]


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.counts)


def redact_pii(text: str) -> RedactionResult:
    counts: dict[str, int] = {}
    out, n = _redact_cards(text)
    if n:
        counts["CARD"] = n
    for label, rx in _PII_REGEXES:
        if label == "ACCOUNT_NO":
            out, n = rx.subn(lambda m: f"{m.group(1)}[REDACTED_ACCOUNT_NO]", out)
        else:
            out, n = rx.subn(f"[REDACTED_{label}]", out)
        if n:
            counts[label] = counts.get(label, 0) + n
    return RedactionResult(out, counts)


# ------------------------------------------------------------------------- output
# Phrases that turn guidance into regulated advice / promissory language.
ADVICE_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern)
    for name, pattern in {
        "direct_recommendation": (
            r"\bi (strongly |would |'d )?(recommend|suggest|advise)\b.{0,40}"
            r"\b(buy|sell|invest|put|move|transfer|choose|open|switch)\b"
        ),
        "imperative_you_should": (
            r"\byou (should|must|ought to|need to|have to)\b.{0,40}"
            r"\b(buy|sell|invest in|put (your|all)|transfer|switch to|choose)\b"
        ),
        "guaranteed_returns": (
            r"(?<!not )(?<!no )(?<!never )\bguarantee[d]? (returns?|growth|profits?)\b"
            r"|\brisk[- ]free (return|investment|growth)"
        ),
        "certainty_claim": r"\bwill (definitely|certainly|surely) (grow|rise|increase|outperform)\b",
        # Fairness heuristic: steering on a protected characteristic.
        "protected_characteristic_steering": (
            r"\bbecause you(?:'re| are) (?:a |an )?"
            r"(woman|man|female|male|old|young|elderly|disabled|retired)\b"
        ),
    }.items()
}


@dataclass
class OutputVerdict:
    status: str  # PASSED | PASSED_PII_REDACTED | BLOCKED_UNSAFE_ADVICE
    text: str
    reasons: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.status.startswith("BLOCKED")


def finalize_output(text: str, sources: list[str] | None = None) -> OutputVerdict:
    """Deterministic egress filter: block advice, redact leaked PII, guarantee the
    citation footer and the FCA notice. Runs on every model answer (no extra LLM call,
    so it cannot rewrite or drop facts)."""
    norm = normalise(text)
    hits = [n for n, rx in ADVICE_PATTERNS.items() if rx.search(norm)]
    if hits:
        return OutputVerdict("BLOCKED_UNSAFE_ADVICE", OUTPUT_BLOCK_MESSAGE, hits)

    red = redact_pii(text)
    body = red.text.rstrip()
    if sources and "sources:" not in body.lower():
        body += "\n\n**Sources:** " + "; ".join(sources)
    if "Regulatory Notice" not in body:
        body += DISCLAIMER
    status = "PASSED_PII_REDACTED" if red.changed else "PASSED"
    return OutputVerdict(status, body, [f"pii:{k}" for k in red.counts])
