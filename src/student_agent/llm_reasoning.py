"""LLM-powered reasoning for specialist agents.

Each specialist calls Qwen 3 8B (via OpenRouter) to produce a structured
analysis of the MCP evidence it collected.  The LLM output is used as an
*advisory* reasoning layer — the deterministic rules in workflow.py remain
the authoritative decision-maker so that verification invariants are never
violated.

The reasoning summaries are recorded in the trace as attributes, making the
LLM's contribution fully observable and auditable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts per specialist role
# ---------------------------------------------------------------------------

_ORDER_SYSTEM = """\
You are an Order Analysis Agent in a multi-agent e-commerce dispute system.
You receive order and order-item evidence from an MCP server.
Analyze the data and produce a JSON object with these fields:
- "order_status": the order status string
- "has_items": boolean, whether order items were found
- "item_count": integer, number of unique order items
- "total_value": float or null, sum of (price + freight) across items
- "seller_ids": list of unique seller IDs found
- "anomalies": list of strings describing any data issues found
Keep the analysis factual and concise.  Output ONLY valid JSON."""

_PAYMENT_SYSTEM = """\
You are a Payment Analysis Agent in a multi-agent e-commerce dispute system.
You receive payment records, payment timelines, and/or refund timelines from an MCP server.
Analyze the data and produce a JSON object with these fields:
- "has_positive_payment": boolean
- "payment_count": integer
- "total_paid": float or null
- "captured_total": float or null
- "capture_count": integer
- "has_mismatch": boolean, whether reconciliation mismatches exist
- "refund_statuses": list of refund status strings found
- "anomalies": list of strings describing any data issues found
Keep the analysis factual and concise.  Output ONLY valid JSON."""

_SHIPMENT_SYSTEM = """\
You are a Shipment Analysis Agent in a multi-agent e-commerce dispute system.
You receive shipment summary evidence from an MCP server.
Analyze the data and produce a JSON object with these fields:
- "was_delivered": boolean
- "delivered_on_time": boolean or null (null if cannot determine)
- "late_actor": string or null ("seller", "logistics_provider", or null)
- "estimated_delivery": string or null (ISO timestamp)
- "actual_delivery": string or null (ISO timestamp)
- "anomalies": list of strings describing any data issues found
Keep the analysis factual and concise.  Output ONLY valid JSON."""

_POLICY_SYSTEM = """\
You are a Policy Analysis Agent in a multi-agent e-commerce dispute system.
You receive policy rules from an MCP server that define how disputes should be resolved.
Analyze the policy data and produce a JSON object with these fields:
- "available_rules": list of rule keys present in the policy
- "rule_count": integer
- "has_refund_rules": boolean
- "anomalies": list of strings describing any data issues found
Keep the analysis factual and concise.  Output ONLY valid JSON."""

_COORDINATOR_SYSTEM = """\
You are the Coordinator Agent in a multi-agent e-commerce dispute resolution system.
You receive a customer claim and evidence summaries from specialist agents.
Synthesize the information and produce a JSON object with these fields:
- "primary_topic": the customer's primary claim topic
- "evidence_supports_claim": boolean, whether MCP evidence corroborates the claim
- "suggested_issue": string, the issue code that best matches the evidence
- "confidence_assessment": string, one of "high", "medium", "low"
- "reasoning": a brief (2-3 sentence) explanation of your assessment
Keep the analysis factual.  Output ONLY valid JSON."""

ROLE_PROMPTS = {
    "order-item-agent": _ORDER_SYSTEM,
    "payment-agent": _PAYMENT_SYSTEM,
    "shipment-agent": _SHIPMENT_SYSTEM,
    "policy-agent": _POLICY_SYSTEM,
    "coordinator": _COORDINATOR_SYSTEM,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

async def analyze_evidence(
    llm: LLMClient,
    *,
    role: str,
    evidence: dict[str, dict[str, Any]],
    case_context: str = "",
) -> dict[str, Any]:
    """Ask the LLM to analyze evidence for a given specialist role.

    Returns the parsed JSON analysis, or an error dict if the LLM call fails.
    """
    system_prompt = ROLE_PROMPTS.get(role)
    if system_prompt is None:
        return {"error": f"no prompt defined for role {role}"}

    # Build a compact user message with the evidence
    evidence_summary = {}
    for tool_name, tool_evidence in evidence.items():
        data = tool_evidence.get("data")
        if data is not None:
            evidence_summary[tool_name] = data

    user_message = f"Case context: {case_context}\n\nEvidence:\n{json.dumps(evidence_summary, ensure_ascii=False, default=str)}"

    try:
        result = await asyncio.wait_for(
            llm.chat_json(
                system=system_prompt,
                user=user_message,
                max_tokens=1024,
            ),
            timeout=8.0,
        )
        return result
    except Exception as exc:
        logger.debug("LLM analysis failed for %s: %s", role, exc)
        return {"error": str(exc), "role": role}


async def synthesize_assessment(
    llm: LLMClient,
    *,
    case: dict[str, Any],
    specialist_analyses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Ask the LLM coordinator to synthesize specialist analyses into a case assessment."""
    system_prompt = ROLE_PROMPTS["coordinator"]

    request = case.get("customer_request", {})
    claims = request.get("claims", []) if isinstance(request, dict) else []
    primary_topic = claims[0].get("topic", "") if claims and isinstance(claims[0], dict) else ""
    customer_message = request.get("message", "") if isinstance(request, dict) else ""

    user_message = (
        f"Customer claim topic: {primary_topic}\n"
        f"Customer message: {customer_message[:500]}\n\n"
        f"Specialist analyses:\n{json.dumps(specialist_analyses, ensure_ascii=False, default=str)}"
    )

    try:
        result = await asyncio.wait_for(
            llm.chat_json(
                system=system_prompt,
                user=user_message,
                max_tokens=1024,
            ),
            timeout=8.0,
        )
        return result
    except Exception as exc:
        logger.debug("LLM synthesis failed: %s", exc)
        return {"error": str(exc)}

