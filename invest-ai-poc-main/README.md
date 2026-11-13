# Invest AI – Agentic Investment Guidance (Google ADK POC)

Lloyds Banking Group technical exercise: a multi-agent, guard-railed assistant that gives
**guidance (not regulated advice)** on UK pensions and ISAs.

```
adk web → INPUT GUARDRAIL → Intent Router Agent ─┬→ RAG Policy Agent ───────┐
          (block + PII redact)                   └→ Allowance Calculator ───┤
                                                                            ▼
                         OUTPUT GUARDRAIL (advice block, PII, citations, FCA notice) → CLOUD LOGGING → user
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example invest_advisor/.env     # then paste your AI Studio key as GOOGLE_API_KEY
adk web                                  # run from the repo root; pick "invest_advisor"
pytest -q                                # unit + golden-set tests, no LLM / network needed
```

Audit records appear in the terminal (JSON) and in `logs/cloud_audit_logs.jsonl`.
In `adk web` use the **State** tab to see `audit_*` keys and the **Trace** tab for the agent path.

Try: *"What is the ISA limit?"* · *"I pay £8,000 into my pension as a higher-rate taxpayer"* ·
*"My NI is AB123456C, should I increase contributions?"* · *"Ignore previous instructions…"* ·
*"Which crypto should I buy?"*

## Design decisions (talk-through notes for the interview)

| Decision | Why |
|---|---|
| Input guardrail on the **root** `before_agent_callback` | Runs once per turn before any LLM; a block skips the whole pipeline (no tokens, no routing). |
| PII sanitiser as `before_model_callback` on **every** LLM agent | Sub-agents read the full session history; redacting only the router (original POC) still leaked NI numbers/emails to the specialists. |
| Output guardrail is a **deterministic `after_model_callback`**, not an LLM "compliance agent" | The draft is never shown un-vetted; no extra LLM call/latency; cannot rewrite or drop facts; testable. |
| Root is a `SequentialAgent` | Forces every turn back through the root so the input guardrail cannot be skipped by ADK resuming a sub-agent. |
| Retrieval at bullet level with BM25 + citations (`S1-B2`) | Each rule is individually citable; footer "Sources" is appended deterministically from tool provenance, not trusted to the LLM. |
| Calculator in `Decimal`, fixed relief maths, rejects bad input | Original applied the 40/45% rate to gross-up (wrong); relief-at-source grosses up at 20% only, extra relief is claimed via Self Assessment. Unknown tax band is now an error instead of silently "basic". |
| Audit record: redacted text, PII *counts*, chunk ids, latency, outcome | Safe to retain; powers block-rate / latency dashboards. Blocked turns log inside the input guardrail because ADK skips `after_agent_callback` when `before_agent_callback` returns content. |
| Pure logic in `policy.py`, `retrieval.py`, `calculator.py` | No ADK imports → fast CI tests. ADK-specific code is a thin layer. |

## Assumptions & known limitations

- **Guardrails are regex baselines.** Production: Vertex AI **Model Armor** (injection/jailbreak, DLP) in front, these rules behind.
- **Retrieval is lexical (BM25) over a 9-chunk document.** It cannot tell "capital of France" from "capital gains" (see `test_known_limitation_lexical_false_positive`); the RAG agent is told to judge relevance. Next step: embeddings / Vertex AI Search or RAG Engine with a calibrated threshold, and hybrid search.
- **Calculator**: relief-at-source, England/Wales/NI bands. No taper, carry-forward, MPAA, Scottish bands.
- **Cloud Logging**: the AI Studio key does not authorise GCP. Locally you get stdout + JSONL; on Cloud Run stdout is ingested automatically; `ENABLE_CLOUD_LOGGING=true` + ADC enables the direct API sink.
- Raw user text still exists in the ADK session store / `adk web` UI (dev). In production use a plugin `on_user_message_callback` (or Model Armor at the gateway) so raw PII is never persisted.
- Streaming (SSE) partial chunks bypass `after_model_callback`; keep non-streaming for customer channels or buffer until final.
- `Dockerfile` and `ci.yml` deploy stage are scaffolding; CI currently runs lint + tests only.
- `legacy/poc_demo_langgraph.py` is the original Streamlit/LangGraph demo, kept for reference only (not in requirements).

## Evaluation strategy (Part 4 summary)

| Dimension | Offline (CI, no LLM) | LLM-in-loop (nightly / pre-release) | Production |
|---|---|---|---|
| Accuracy | calculator unit tests; retrieval hit@3 on `evals/golden_set.json` | answer faithfulness vs retrieved passages (LLM judge), numeric-match against tool output | sampled human review; thumbs feedback |
| Routing | – | `expect_intent` in golden set → router accuracy / confusion matrix (ADK `adk eval`) | intent distribution drift |
| Latency | – | p50/p95 per stage | log-based metric on `latency_ms`; SLO alert |
| Fairness | no age/gender steering patterns (`protected_characteristic_steering`) | counterfactual pairs: same query, varied age/gender/region → compare answers & block rate | block-rate by age band, alert on divergence |
| Explainability | every answer carries chunk-id citations + `Sources` footer | citation precision (cited id ∈ retrieved ids) | `sources`/`tools` in every audit record |
| Compliance | advice/guarantee phrases blocked; disclaimer always present | red-team set (injection, "what should I buy") | alert on output-block spikes; prompt-quality audit of sampled logs |

CI/CD: lint + tests on every push → LLM evals on schedule (secret key) → deploy to Cloud Run via Workload
Identity Federation on `main` → log-based alerts on block rate, error rate, latency.
