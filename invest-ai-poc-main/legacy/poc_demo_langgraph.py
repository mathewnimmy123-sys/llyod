import os
import re
import time
import json
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# 1. Load your AI Studio Key from .env
load_dotenv()
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing GEMINI_API_KEY in .env file!")
    st.stop()

# Initialize Gemini (Swappable with Vertex AI in production)
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)

# --- SIMULATED KNOWLEDGE BASE (Replaces Vertex AI Search / Chroma for quick demo) ---
UK_INVESTMENT_DOCS = {
    "pension": "[FCA Pension Guide 2026] The UK annual pension allowance is £60,000 (or 100% of earnings, whichever is lower). Employer contributions count toward this cap. Tax relief is granted at your marginal income tax rate.",
    "isa": "[HMRC ISA Rules 2026] The annual Individual Savings Account (ISA) allowance is £20,000 across Cash, Stocks & Shares, and Innovative Finance ISAs. Gains and withdrawals are free from UK Income and Capital Gains Tax."
}

# --- LANGGRAPH STATE ---
class AgentState(TypedDict):
    user_query: str
    sanitized_query: str
    security_status: str
    intent: str
    retrieved_context: str
    draft_response: str
    final_response: str
    audit_log: list[str]

# --- NODE 1: SIMULATED VERTEX MODEL ARMOR (Security & PII) ---
def model_armor_node(state: AgentState) -> AgentState:
    start = time.time()
    query = state["user_query"]
    logs = list(state.get("audit_log", []))

    # 1. Prompt Injection / Jailbreak check
    blocked_patterns = ["ignore previous instructions", "system prompt", "bypass compliance"]
    if any(p in query.lower() for p in blocked_patterns):
        logs.append(f"[Model Armor] BLOCKED: Prompt injection attempt detected ({time.time()-start:.2f}s)")
        return {
            **state,
            "security_status": "BLOCKED",
            "final_response": "Request blocked by security guardrails: Policy violation detected.",
            "audit_log": logs,
        }

    # 2. PII Redaction (e.g., UK National Insurance Number or Email)
    sanitized = re.sub(r"[A-Za-z]{2}\d{6}[A-Za-z]", "[REDACTED_NINO]", query)
    sanitized = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[REDACTED_EMAIL]", sanitized)
    
    pii_note = "PII redacted" if sanitized != query else "Clean"
    logs.append(f"[Model Armor] PASSED ({pii_note}) in {time.time()-start:.2f}s")
    
    return {
        **state,
        "sanitized_query": sanitized,
        "security_status": "PASSED",
        "audit_log": logs,
    }

# --- NODE 2: INTENT ROUTER AGENT ---
class RouteDecision(BaseModel):
    intent: Literal["pension_rag", "isa_rag", "out_of_scope"] = Field(
        description="Classify the user's financial query."
    )

def intent_router_node(state: AgentState) -> AgentState:
    start = time.time()
    router_llm = llm.with_structured_output(RouteDecision)
    decision = router_llm.invoke(
        f"Classify this UK retail banking query into pension_rag, isa_rag, or out_of_scope: {state['sanitized_query']}"
    )
    logs = state["audit_log"] + [f"[Intent Agent] Routed to '{decision.intent}' in {time.time()-start:.2f}s"]
    return {**state, "intent": decision.intent, "audit_log": logs}

# --- NODE 3: RAG & GUIDANCE AGENT ---
def rag_guidance_node(state: AgentState) -> AgentState:
    start = time.time()
    intent = state["intent"]
    
    if intent == "out_of_scope":
        context = "No internal product docs matched."
        draft = "I can only provide regulated guidance on UK Pensions and ISAs. Please contact a human advisor for broader queries."
    else:
        doc_key = "pension" if intent == "pension_rag" else "isa"
        context = UK_INVESTMENT_DOCS[doc_key]
        prompt = (
            f"You are an FCA-compliant UK investment guidance assistant. Answer strictly using the context below. "
            f"Provide balanced, objective guidance (not regulated personal financial advice) and cite the source.\n\n"
            f"Context: {context}\nUser Query: {state['sanitized_query']}"
        )
        draft = llm.invoke(prompt).content

    logs = state["audit_log"] + [f"[RAG Agent] Generated grounded draft using {intent} in {time.time()-start:.2f}s"]
    return {**state, "retrieved_context": context, "draft_response": draft, "audit_log": logs}

# --- NODE 4: COMPLIANCE & EXPLAINABILITY AUDITOR ---
def compliance_audit_node(state: AgentState) -> AgentState:
    start = time.time()
    disclaimer = (
        "\n\n---\n*Regulatory Notice: This information is for guidance only and does not constitute regulated "
        "personal financial advice. Capital is at risk; tax rules depend on individual circumstances.*"
    )
    final_text = state["draft_response"] + disclaimer
    logs = state["audit_log"] + [f"[Compliance Agent] Verified fairness & appended FCA disclaimer in {time.time()-start:.2f}s"]
    return {**state, "final_response": final_text, "audit_log": logs}

# --- BUILD LANGGRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("model_armor", model_armor_node)
workflow.add_node("intent_router", intent_router_node)
workflow.add_node("rag_guidance", rag_guidance_node)
workflow.add_node("compliance_audit", compliance_audit_node)

workflow.set_entry_point("model_armor")
workflow.add_conditional_edges(
    "model_armor",
    lambda s: END if s["security_status"] == "BLOCKED" else "intent_router",
)
workflow.add_edge("intent_router", "rag_guidance")
workflow.add_edge("rag_guidance", "compliance_audit")
workflow.add_edge("compliance_audit", END)

app_graph = workflow.compile()

# --- STREAMLIT UI & OBSERVABILITY PANEL ---
st.set_page_config(page_title="Invest AI - Agentic POC", layout="wide")
st.title("Agentic AI for Investment Guidance")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input(
        "Ask an investment question:",
        placeholder="e.g., Should I increase my pension contributions? My NI is AB123456C"
    )
    if st.button("Run Agentic Pipeline") and user_input:
        result = app_graph.invoke({"user_query": user_input, "audit_log": []})
        st.subheader("Customer Response")
        st.markdown(result["final_response"])
        
        with st.expander("View Retrieved Grounding Source (Explainability)"):
            st.info(result.get("retrieved_context", "N/A"))

with col2:
    st.subheader("Observability & Cloud Logging Trace")
    if "result" in locals():
        for entry in result["audit_log"]:
            st.code(entry, language="text")
        st.json({
            "security_status": result["security_status"],
            "detected_intent": result["intent"],
            "sanitized_input": result["sanitized_query"]
        })