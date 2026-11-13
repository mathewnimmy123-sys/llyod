"""Offline golden-set checks: guardrail block decisions and retrieval hit@3. No LLM needed."""

import json
from pathlib import Path

import pytest

from invest_advisor import policy
from invest_advisor.retrieval import Retriever, load_chunks

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads((ROOT / "evals" / "golden_set.json").read_text())["cases"]
DOC = ROOT / "data" / "uk_investment_guidance_2026.txt"


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_chunks(DOC), DOC.name)


def test_document_chunking(retriever):
    ids = [c.chunk_id for c in retriever.chunks]
    assert ids == ["S1-B1", "S1-B2", "S1-B3", "S1-B4", "S2-B1", "S2-B2", "S2-B3", "S3-B1", "S3-B2"]


@pytest.mark.parametrize("case", CASES, ids=[c["query"][:40] for c in CASES])
def test_guardrail_decision(case):
    assert policy.check_input(case["query"]).blocked is case["expect_blocked"]


@pytest.mark.parametrize(
    "case", [c for c in CASES if c["expect_chunk"]], ids=lambda c: c["query"][:40]
)
def test_retrieval_hit_at_3(case, retriever):
    res = retriever.search(case["query"])
    assert res["status"] == "success"
    assert case["expect_chunk"] in [p["chunk_id"] for p in res["passages"]]


@pytest.mark.parametrize("q", ["What is the weather in London?", "Tell me a joke"])
def test_retrieval_abstains_when_no_term_overlap(q, retriever):
    assert retriever.search(q)["status"] == "no_match"


def test_known_limitation_lexical_false_positive(retriever):
    """BM25 on a tiny corpus cannot tell 'capital of France' from 'capital gains'. This is
    why the RAG agent is instructed to judge passage relevance itself, and why the
    production plan is embeddings / Vertex AI Search with a calibrated threshold."""
    assert retriever.search("What is the capital of France?")["status"] == "success"
