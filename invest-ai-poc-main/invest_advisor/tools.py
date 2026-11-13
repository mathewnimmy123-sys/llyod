"""ADK tool wrappers. The logic lives in retrieval.py / calculator.py (pure + tested);
these functions only adapt them to ADK (docstrings become the tool schema the LLM sees)
and record provenance in session state for explainability and audit."""

from functools import lru_cache
from typing import Optional

from google.adk.tools import ToolContext

from . import config
from .calculator import CalculationError, calculate_relief
from .retrieval import Retriever, load_chunks


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return Retriever(load_chunks(config.DATA_FILE), config.DATA_FILE.name)


def _note(tool_context: Optional[ToolContext], tool: str, sources: list) -> None:
    if tool_context is None:
        return
    st = tool_context.state
    st["audit_tools"] = list(st.get("audit_tools") or []) + [tool]
    merged = list(st.get("audit_sources") or [])
    st["audit_sources"] = merged + [s for s in sources if s not in merged]


def search_investment_docs(query: str, tool_context: ToolContext) -> dict:
    """Searches the internal UK pension / ISA / FCA policy document and returns the most
    relevant passages with citable chunk ids (e.g. S1-B2).

    Args:
        query: The customer's topic in plain words (e.g. 'ISA annual limit', 'pension tax relief').

    Returns:
        status 'success' with `passages` (chunk_id, section, text), or 'no_match' when the
        document does not cover the topic - in which case do not answer from general knowledge.
    """
    if not config.DATA_FILE.exists():
        return {"status": "error", "message": "Policy document is unavailable."}
    result = _retriever().search(query)
    sources = [f"{result['source_document']} ({p['chunk_id']})" for p in result["passages"]]
    _note(tool_context, "search_investment_docs", sources)
    return result


def calculate_pension_tax_relief(
    contribution_gbp: float,
    tax_band: str = "basic",
    employer_contribution_gbp: float = 0.0,
    annual_earnings_gbp: float = 0.0,
    tool_context: ToolContext = None,  # injected by ADK; not part of the LLM schema
) -> dict:
    """Calculates estimated UK pension tax relief and remaining £60,000 annual allowance.

    Args:
        contribution_gbp: Amount the customer pays in from their own net pay, in GBP.
        tax_band: 'basic' (20%), 'higher' (40%) or 'additional' (45%). Ask the customer if unknown.
        employer_contribution_gbp: Employer contribution for the year in GBP (0 if none/unknown).
        annual_earnings_gbp: Relevant UK annual earnings in GBP (0 if unknown).

    Returns:
        A dict of exact figures plus `assumptions`; or status 'error' with a `message`.
    """
    try:
        result = calculate_relief(contribution_gbp, tax_band, employer_contribution_gbp, annual_earnings_gbp)
    except CalculationError as exc:
        return {"status": "error", "message": str(exc)}
    _note(tool_context, "calculate_pension_tax_relief",
          ["Allowance calculator (relief-at-source, £60,000 annual allowance)"])
    return result
