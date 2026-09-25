from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from .llm_client import LLMClient
from .llm_reasoning import analyze_evidence, synthesize_assessment
from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

logger = logging.getLogger(__name__)

ORDER_AGENT = "order-item-agent"
PAYMENT_AGENT = "payment-agent"
SHIPMENT_AGENT = "shipment-agent"
POLICY_AGENT = "policy-agent"
VERIFIER_AGENT = "verifier-agent"

ISSUE_INVARIANTS = {
    "canceled_order_paid": ("action_required", "platform", "issue_refund"),
    "unavailable_order_paid": ("action_required", "seller", "issue_refund"),
    "late_delivery_seller": ("action_required", "seller", "refund_freight"),
    "late_delivery_logistics": (
        "action_required",
        "logistics_provider",
        "refund_freight",
    ),
    "valid_split_payment": ("no_action", "customer", "document_no_action"),
    "payment_mismatch": ("action_required", "payment_provider", "reconcile_payment"),
    "duplicate_charge": (
        "action_required",
        "payment_provider",
        "refund_duplicate_charge",
    ),
    "refund_pending": ("needs_investigation", "payment_provider", "monitor_refund"),
    "refund_failed": ("action_required", "payment_provider", "retry_refund"),
    "unsupported_claim": ("no_action", "customer", "document_no_action"),
    "insufficient_evidence": (
        "needs_investigation",
        "unknown",
        "collect_authoritative_evidence",
    ),
}

TOOL_PERMISSIONS = {
    ORDER_AGENT: frozenset({"get_order", "get_order_items", "get_sellers"}),
    PAYMENT_AGENT: frozenset(
        {"get_order_payments", "get_payment_timeline", "get_refund_timeline"}
    ),
    SHIPMENT_AGENT: frozenset({"get_shipment_summary"}),
    POLICY_AGENT: frozenset({"get_policy"}),
}

ITEM_TOPICS = {
    "unavailable_order_paid",
    "late_delivery_seller",
    "late_delivery_logistics",
    "valid_split_payment",
    "payment_mismatch",
    "duplicate_charge",
    "unsupported_claim",
}
TIMELINE_TOPICS = {
    "valid_split_payment",
    "payment_mismatch",
    "duplicate_charge",
    "unsupported_claim",
}
REFUND_TOPICS = {"refund_pending", "refund_failed"}
SHIPMENT_TOPICS = {
    "late_delivery_seller",
    "late_delivery_logistics",
    "unsupported_claim",
}


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: dict[str, str]


@dataclass
class AgentReport:
    actor: str
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _data(evidence_by_tool: dict[str, dict[str, Any]], tool_name: str) -> Any:
    evidence = evidence_by_tool.get(tool_name, {})
    return evidence.get("data")


def _near(left: Any, right: Any, days: float = 2.0) -> bool:
    left_time = _parse_time(left)
    right_time = _parse_time(right)
    if left_time is None or right_time is None:
        return False
    return abs((left_time - right_time).total_seconds()) <= days * 86_400


def _selected_items(evidence_by_tool: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    raw_items = _data(evidence_by_tool, "get_order_items")
    if not isinstance(raw_items, list):
        return []
    order = _data(evidence_by_tool, "get_order")
    purchase_at = order.get("order_purchase_timestamp") if isinstance(order, dict) else None
    purchase_time = _parse_time(purchase_at)
    selected: dict[str, tuple[float, dict[str, Any]]] = {}
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        item_id = item.get("order_item_id")
        if not isinstance(item_id, str):
            continue
        shipping_time = _parse_time(item.get("shipping_limit_date"))
        distance = float(index)
        if purchase_time is not None and shipping_time is not None:
            distance = abs((shipping_time - purchase_time).total_seconds())
        current = selected.get(item_id)
        if current is None or distance < current[0]:
            selected[item_id] = (distance, item)
    return [selected[item_id][1] for item_id in sorted(selected)]


def _expected_total(evidence_by_tool: dict[str, dict[str, Any]]) -> Decimal | None:
    items = _selected_items(evidence_by_tool)
    if not items:
        return None
    total = Decimal("0")
    for item in items:
        price = _decimal(item.get("price"))
        freight = _decimal(item.get("freight_value"))
        if price is None or freight is None:
            return None
        total += price + freight
    return total


def _relevant_payment_events(
    evidence_by_tool: dict[str, dict[str, Any]], event_type: str | None = None
) -> list[dict[str, Any]]:
    timeline = _data(evidence_by_tool, "get_payment_timeline")
    order = _data(evidence_by_tool, "get_order")
    if not isinstance(timeline, dict) or not isinstance(timeline.get("events"), list):
        return []
    purchase_at = order.get("order_purchase_timestamp") if isinstance(order, dict) else None
    candidates = [event for event in timeline["events"] if isinstance(event, dict)]
    if event_type is not None:
        candidates = [event for event in candidates if event.get("event_type") == event_type]
    scoped = [event for event in candidates if _near(event.get("event_at"), purchase_at)]
    return scoped or candidates


def _captured_total(evidence_by_tool: dict[str, dict[str, Any]]) -> tuple[Decimal, int]:
    captured = [
        event
        for event in _relevant_payment_events(evidence_by_tool, "captured")
        if event.get("status") == "confirmed"
    ]
    amounts = [_decimal(event.get("amount_brl")) for event in captured]
    valid_amounts = [amount for amount in amounts if amount is not None]
    return sum(valid_amounts, Decimal("0")), len(valid_amounts)


def _has_positive_payment(evidence_by_tool: dict[str, dict[str, Any]]) -> bool:
    payments = _data(evidence_by_tool, "get_order_payments")
    if not isinstance(payments, list):
        return False
    return any(
        isinstance(payment, dict)
        and (amount := _decimal(payment.get("payment_value"))) is not None
        and amount > 0
        for payment in payments
    )


def _refund_statuses(evidence_by_tool: dict[str, dict[str, Any]]) -> set[str]:
    timeline = _data(evidence_by_tool, "get_refund_timeline")
    if not isinstance(timeline, dict) or not isinstance(timeline.get("events"), list):
        return set()
    return {
        status
        for event in timeline["events"]
        if isinstance(event, dict) and isinstance((status := event.get("status")), str)
    }


def _late_actor(evidence_by_tool: dict[str, dict[str, Any]]) -> str | None:
    shipment = _data(evidence_by_tool, "get_shipment_summary")
    if not isinstance(shipment, dict):
        return None
    delivered = _parse_time(shipment.get("delivered_customer_at"))
    estimated = _parse_time(shipment.get("estimated_delivery_at"))
    if delivered is None or estimated is None or delivered <= estimated:
        return None
    events = shipment.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        if (
            isinstance(event, dict)
            and event.get("event_type") == "delivered_late"
            and event.get("status") == "confirmed"
            and _near(event.get("event_at"), shipment.get("delivered_customer_at"), days=1)
            and event.get("actor") in {"seller", "logistics_provider"}
        ):
            return str(event["actor"])
    return None


def _supports_topic(topic: str, evidence_by_tool: dict[str, dict[str, Any]]) -> bool:
    order = _data(evidence_by_tool, "get_order")
    order_status = order.get("order_status") if isinstance(order, dict) else None
    expected_total = _expected_total(evidence_by_tool)
    captured_total, capture_count = _captured_total(evidence_by_tool)
    mismatch_events = _relevant_payment_events(evidence_by_tool, "reconciliation_mismatch")
    if topic == "canceled_order_paid":
        return order_status == "canceled" and _has_positive_payment(evidence_by_tool)
    if topic == "unavailable_order_paid":
        return order_status == "unavailable" and _has_positive_payment(evidence_by_tool)
    if topic == "late_delivery_seller":
        return _late_actor(evidence_by_tool) == "seller"
    if topic == "late_delivery_logistics":
        return _late_actor(evidence_by_tool) == "logistics_provider"
    if topic == "payment_mismatch":
        return any(event.get("status") == "open" for event in mismatch_events)
    if topic == "duplicate_charge":
        return (
            expected_total is not None
            and capture_count > 1
            and captured_total > expected_total
        )
    if topic == "valid_split_payment":
        return (
            expected_total is not None
            and capture_count > 1
            and captured_total == expected_total
            and not mismatch_events
        )
    if topic == "refund_pending":
        return "pending" in _refund_statuses(evidence_by_tool)
    if topic == "refund_failed":
        return "failed" in _refund_statuses(evidence_by_tool)
    if topic == "unsupported_claim":
        shipment = _data(evidence_by_tool, "get_shipment_summary")
        delivered = _parse_time(shipment.get("delivered_customer_at")) if isinstance(
            shipment, dict
        ) else None
        estimated = _parse_time(shipment.get("estimated_delivery_at")) if isinstance(
            shipment, dict
        ) else None
        delivered_on_time = (
            delivered is not None and estimated is not None and delivered <= estimated
        )
        payment_ok = expected_total is not None and captured_total <= expected_total
        return order_status == "delivered" and delivered_on_time and payment_ok
    return False


def _tool_plan(topic: str, order_id: str, policy_version: str) -> dict[str, list[ToolRequest]]:
    plan = {
        ORDER_AGENT: [ToolRequest("get_order", {"order_id": order_id})],
        PAYMENT_AGENT: [],
        SHIPMENT_AGENT: [],
        POLICY_AGENT: [ToolRequest("get_policy", {"policy_version": policy_version})],
    }
    if topic in ITEM_TOPICS:
        plan[ORDER_AGENT].append(ToolRequest("get_order_items", {"order_id": order_id}))
    if topic in {"canceled_order_paid", "unavailable_order_paid"} | REFUND_TOPICS:
        plan[PAYMENT_AGENT].append(ToolRequest("get_order_payments", {"order_id": order_id}))
    if topic in TIMELINE_TOPICS:
        plan[PAYMENT_AGENT].append(ToolRequest("get_payment_timeline", {"order_id": order_id}))
    if topic in REFUND_TOPICS:
        plan[PAYMENT_AGENT].append(ToolRequest("get_refund_timeline", {"order_id": order_id}))
    if topic in SHIPMENT_TOPICS:
        plan[SHIPMENT_AGENT].append(ToolRequest("get_shipment_summary", {"order_id": order_id}))
    return {actor: requests for actor, requests in plan.items() if requests}


async def _call_with_retry(
    gateway: EvidenceGateway,
    request: ToolRequest,
    *,
    case_id: str,
    attempts: int = 2,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await gateway.call(request.name, case_id=case_id, **request.arguments)
        except Exception as exc:  # Network/remote failures vary by MCP transport implementation.
            last_error = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(0.25 * (attempt + 1))
    assert last_error is not None
    raise last_error


async def _run_specialist(
    *,
    actor: str,
    requests: list[ToolRequest],
    case_id: str,
    available_tools: set[str],
    gateway: EvidenceGateway,
    trace: TraceWriter,
    llm: LLMClient | None = None,
) -> AgentReport:
    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target=actor,
        decision_code="COLLECT_SCOPED_EVIDENCE",
        attributes={"tool_count": len(requests)},
    )
    report = AgentReport(actor=actor)
    permissions = TOOL_PERMISSIONS[actor]
    for request in requests:
        if request.name not in permissions:
            report.errors[request.name] = "tool_not_permitted"
            continue
        if request.name not in available_tools:
            report.errors[request.name] = "tool_not_discovered"
            continue
        try:
            evidence = await _call_with_retry(gateway, request, case_id=case_id)
        except Exception as exc:
            report.errors[request.name] = type(exc).__name__
            continue
        report.evidence[request.name] = evidence
        trace.emit(
            case_id=case_id,
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=request.name,
            evidence_refs=[evidence["evidence_ref"]],
        )

    # --- LLM reasoning layer (advisory, non-blocking) ---
    llm_analysis: dict[str, Any] | None = None
    if llm is not None and report.evidence:
        try:
            llm_analysis = await analyze_evidence(
                llm,
                role=actor,
                evidence=report.evidence,
                case_context=f"case_id={case_id}",
            )
            logger.info("LLM analysis for %s/%s: %s", actor, case_id, llm_analysis)
        except Exception as exc:
            logger.warning("LLM analysis skipped for %s/%s: %s", actor, case_id, exc)

    decision_code = "EVIDENCE_READY" if not report.errors else "PARTIAL_EVIDENCE"
    handoff_attrs: dict[str, Any] = {
        "evidence_count": len(report.evidence),
        "error_count": len(report.errors),
    }
    if llm_analysis is not None and "error" not in llm_analysis:
        handoff_attrs["llm_assisted"] = True

    if actor == POLICY_AGENT:
        trace.emit(
            case_id=case_id,
            event_type="policy_decided",
            actor=actor,
            target="coordinator",
            decision_code=decision_code,
            evidence_refs=[item["evidence_ref"] for item in report.evidence.values()],
        )
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=actor,
        target="coordinator",
        decision_code=decision_code,
        attributes=handoff_attrs,
    )
    return report


def _claim_verdict(topic: str, issue: str) -> str:
    if issue == "insufficient_evidence":
        return "insufficient_evidence"
    if topic == "requested_full_refund":
        if issue in {"canceled_order_paid", "unavailable_order_paid", "refund_failed"}:
            return "supported"
        if issue in {
            "late_delivery_seller",
            "late_delivery_logistics",
            "payment_mismatch",
            "duplicate_charge",
            "refund_pending",
        }:
            return "partially_supported"
        return "unsupported"
    if topic == "unsupported_claim":
        return "unsupported"
    return "supported" if topic == issue else "unsupported"


def _policy_rule(
    evidence_by_tool: dict[str, dict[str, Any]], issue: str
) -> dict[str, Any] | None:
    policy = _data(evidence_by_tool, "get_policy")
    if not isinstance(policy, dict) or not isinstance(policy.get("rules"), dict):
        return None
    rule = policy["rules"].get(issue)
    return rule if isinstance(rule, dict) else None


def _claim_evidence_refs(
    topic: str,
    issue: str,
    evidence_by_tool: dict[str, dict[str, Any]],
) -> list[str]:
    """Return only evidence that materially supports this individual claim."""
    if issue == "insufficient_evidence":
        tool_names = tuple(evidence_by_tool)
    elif topic == "requested_full_refund":
        financial_tools = {
            "canceled_order_paid": ("get_order", "get_order_payments", "get_policy"),
            "unavailable_order_paid": ("get_order", "get_order_payments", "get_policy"),
            "late_delivery_seller": ("get_shipment_summary", "get_policy"),
            "late_delivery_logistics": ("get_shipment_summary", "get_policy"),
            "valid_split_payment": ("get_payment_timeline", "get_policy"),
            "payment_mismatch": ("get_payment_timeline", "get_policy"),
            "duplicate_charge": ("get_payment_timeline", "get_policy"),
            "refund_pending": ("get_refund_timeline", "get_policy"),
            "refund_failed": ("get_refund_timeline", "get_policy"),
            "unsupported_claim": ("get_order", "get_payment_timeline", "get_policy"),
        }
        tool_names = financial_tools.get(issue, ("get_policy",))
    else:
        issue_tools = {
            "canceled_order_paid": ("get_order", "get_order_payments", "get_policy"),
            "unavailable_order_paid": (
                "get_order",
                "get_order_items",
                "get_order_payments",
                "get_policy",
            ),
            "late_delivery_seller": (
                "get_order",
                "get_order_items",
                "get_shipment_summary",
                "get_policy",
            ),
            "late_delivery_logistics": (
                "get_order",
                "get_order_items",
                "get_shipment_summary",
                "get_policy",
            ),
            "valid_split_payment": (
                "get_order_items",
                "get_payment_timeline",
                "get_policy",
            ),
            "payment_mismatch": ("get_payment_timeline", "get_policy"),
            "duplicate_charge": (
                "get_order_items",
                "get_payment_timeline",
                "get_policy",
            ),
            "refund_pending": ("get_refund_timeline", "get_policy"),
            "refund_failed": ("get_refund_timeline", "get_policy"),
            "unsupported_claim": (
                "get_order",
                "get_order_items",
                "get_payment_timeline",
                "get_shipment_summary",
                "get_policy",
            ),
        }
        tool_names = issue_tools.get(issue, tuple(evidence_by_tool))
    return list(
        dict.fromkeys(
            evidence_by_tool[tool_name]["evidence_ref"]
            for tool_name in tool_names
            if tool_name in evidence_by_tool
        )
    )


def _detect_data_conflicts(
    evidence_by_tool: dict[str, dict[str, Any]],
    *,
    primary_topic: str,
    issue: str,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    raw_items = _data(evidence_by_tool, "get_order_items")
    selected_items = {
        item["order_item_id"]: item
        for item in _selected_items(evidence_by_tool)
        if isinstance(item.get("order_item_id"), str)
    }
    if isinstance(raw_items, list):
        rows_by_item: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, item in enumerate(raw_items):
            if isinstance(item, dict) and isinstance(item.get("order_item_id"), str):
                rows_by_item.setdefault(item["order_item_id"], []).append((index, item))
        for item_id, rows in rows_by_item.items():
            fingerprints = {
                (
                    row.get("seller_id"),
                    row.get("shipping_limit_date"),
                    row.get("price"),
                    row.get("freight_value"),
                )
                for _, row in rows
            }
            if len(fingerprints) <= 1:
                continue
            sources = [f"get_order_items:row_{index + 1}" for index, _ in rows[:5]]
            selected = selected_items.get(item_id)
            selected_source = None
            for index, row in rows:
                if row is selected:
                    selected_source = f"get_order_items:row_{index + 1}"
                    break
            conflicts.append(
                {
                    "field": f"order_items.{item_id}",
                    "sources": sources,
                    "selected_source": selected_source,
                    "resolution_code": "NEAREST_PURCHASE_TIMESTAMP",
                }
            )
            if len(conflicts) == 5:
                return conflicts

    timeline = _data(evidence_by_tool, "get_payment_timeline")
    order = _data(evidence_by_tool, "get_order")
    purchase_at = order.get("order_purchase_timestamp") if isinstance(order, dict) else None
    if isinstance(timeline, dict) and isinstance(timeline.get("events"), list):
        events = [event for event in timeline["events"] if isinstance(event, dict)]
        in_scope = [event for event in events if _near(event.get("event_at"), purchase_at)]
        out_of_scope = [event for event in events if event not in in_scope]
        if in_scope and out_of_scope:
            conflicts.append(
                {
                    "field": "payment_timeline.events",
                    "sources": ["purchase_window", "out_of_scope_events"],
                    "selected_source": "purchase_window",
                    "resolution_code": "CASE_TIME_SCOPE",
                }
            )

    shipment = _data(evidence_by_tool, "get_shipment_summary")
    if isinstance(shipment, dict) and isinstance(shipment.get("events"), list):
        contradictory = any(
            isinstance(event, dict)
            and event.get("event_type") == "delivered_late"
            and not _near(
                event.get("event_at"), shipment.get("delivered_customer_at"), days=1
            )
            for event in shipment["events"]
        )
        if contradictory:
            conflicts.append(
                {
                    "field": "shipment_summary.delivery_timing",
                    "sources": ["shipment_timestamps", "shipment_events"],
                    "selected_source": "shipment_timestamps",
                    "resolution_code": "TIMESTAMP_CONSISTENCY",
                }
            )

    if primary_topic and issue not in {primary_topic, "insufficient_evidence"}:
        conflicts.append(
            {
                "field": "customer_request.claims[0].topic",
                "sources": ["customer_request", "mcp_evidence"],
                "selected_source": "mcp_evidence",
                "resolution_code": "AUTHORITATIVE_EVIDENCE_PREVAILS",
            }
        )
    return conflicts[:5]


def _calibrate_confidence(
    issue: str,
    evidence_by_tool: dict[str, dict[str, Any]],
    errors: dict[str, str],
    conflicts: list[dict[str, Any]],
) -> float:
    successful = len(evidence_by_tool)
    attempted = successful + len(errors)
    completeness = successful / attempted if attempted else 0.0
    warnings = sum(
        len(evidence.get("warnings", []))
        for evidence in evidence_by_tool.values()
        if isinstance(evidence.get("warnings", []), list)
    )
    if issue == "insufficient_evidence":
        base = 0.25 + 0.30 * completeness
    elif issue == "unsupported_claim":
        base = 0.92 * completeness
    else:
        base = 0.96 * completeness
    penalty = min(0.30, 0.04 * len(conflicts) + 0.02 * warnings)
    confidence = max(0.05, min(0.98, base - penalty))
    return round(confidence, 2)


def _build_output(
    case: dict[str, Any],
    evidence_by_tool: dict[str, dict[str, Any]],
    errors: dict[str, str],
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    request = case.get("customer_request", {})
    claims = request.get("claims", []) if isinstance(request, dict) else []
    primary_topic = claims[0].get("topic") if claims and isinstance(claims[0], dict) else ""
    supported = isinstance(primary_topic, str) and _supports_topic(
        primary_topic, evidence_by_tool
    )
    issue = primary_topic if supported else "unsupported_claim"
    if errors or not evidence_by_tool.get("get_policy"):
        issue = "insufficient_evidence"
    rule = _policy_rule(evidence_by_tool, issue)
    if rule is None and issue != "insufficient_evidence":
        issue = "insufficient_evidence"
    rule = _policy_rule(evidence_by_tool, issue) or {}

    evidence_refs = list(
        dict.fromkeys(
            evidence["evidence_ref"]
            for evidence in evidence_by_tool.values()
            if isinstance(evidence.get("evidence_ref"), str)
        )
    )
    items = _selected_items(evidence_by_tool)
    item_ids = list(
        dict.fromkeys(
            str(item["order_item_id"])
            for item in items
            if isinstance(item.get("order_item_id"), str)
        )
    )
    seller_ids = list(
        dict.fromkeys(
            str(item["seller_id"])
            for item in items
            if isinstance(item.get("seller_id"), str)
        )
    )
    order_id = request.get("claimed_order_id") if isinstance(request, dict) else None
    order_ids = [order_id] if isinstance(order_id, str) else []

    case_status = rule.get("case_status", "needs_investigation")
    action = rule.get("recommended_action", "collect_authoritative_evidence")
    refund = _decimal(rule.get("refund_brl")) or Decimal("0")
    responsible = rule.get("responsible_parties", [])
    responsible_parties: list[dict[str, Any]] = []
    if isinstance(responsible, list):
        for party in responsible:
            if not isinstance(party, dict) or not isinstance(party.get("party_type"), str):
                continue
            party_id = party.get("party_id")
            if party["party_type"] == "seller" and seller_ids:
                party_id = seller_ids[0]
            responsible_parties.append(
                {"party_type": party["party_type"], "party_id": party_id}
            )
    if not responsible_parties:
        responsible_parties = [{"party_type": "unknown", "party_id": None}]

    data_conflicts = _detect_data_conflicts(
        evidence_by_tool,
        primary_topic=str(primary_topic),
        issue=str(issue),
    )
    confidence = _calibrate_confidence(issue, evidence_by_tool, errors, data_conflicts)
    claim_assessments = []
    for claim in claims[:5]:
        if not isinstance(claim, dict) or not isinstance(claim.get("claim_id"), str):
            continue
        topic = claim.get("topic", "")
        verdict = _claim_verdict(str(topic), issue)
        claim_confidence = confidence
        if verdict == "partially_supported":
            claim_confidence = round(max(0.05, confidence - 0.08), 2)
        elif verdict == "insufficient_evidence":
            claim_confidence = min(confidence, 0.55)
        claim_assessments.append(
            {
                "claim_id": claim["claim_id"],
                "verdict": verdict,
                "confidence": claim_confidence,
                "evidence_refs": _claim_evidence_refs(
                    str(topic), issue, evidence_by_tool
                ),
            }
        )

    cause_code = issue.upper()
    if issue == "unsupported_claim":
        cause_code = "CLAIM_NOT_CORROBORATED"
    elif issue == "insufficient_evidence":
        cause_code = "MISSING_AUTHORITATIVE_EVIDENCE"
    refund_lines = []
    if refund > 0:
        refund_lines.append(
            {
                "reason_code": str(action),
                "amount_brl": float(refund),
                "entity_id": order_id if isinstance(order_id, str) else None,
            }
        )
    return {
        "schema_version": "day09-l3a-output-v2",
        "case_id": case_id,
        "assessment": {
            "primary_issue": issue,
            "case_status": case_status,
            "confidence": confidence,
        },
        "affected_entities": {
            "order_ids": order_ids,
            "item_ids": item_ids,
            "seller_ids": seller_ids,
            "payment_references": [],
            "shipment_ids": [],
        },
        "claim_assessments": claim_assessments,
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": cause_code, "rank": 1}],
            "responsible_parties": responsible_parties,
        },
        "evidence_refs": evidence_refs,
        "data_conflicts": data_conflicts,
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": float(refund),
            "refund_lines": refund_lines,
        },
        "resolution_actions": [str(action)],
    }


def _verify_output(
    output: dict[str, Any],
    *,
    case_id: str,
    collected_refs: set[str],
    trace: TraceWriter,
) -> None:
    if output.get("case_id") != case_id:
        raise ValueError("verifier rejected mismatched case_id")
    output_refs = set(output.get("evidence_refs", []))
    if not output_refs.issubset(collected_refs):
        raise ValueError("verifier rejected an evidence_ref not collected for this case")
    for claim in output.get("claim_assessments", []):
        if not set(claim.get("evidence_refs", [])).issubset(output_refs):
            raise ValueError("verifier rejected claim evidence outside output evidence_refs")
    financial = output["financial_resolution"]
    line_total = sum(
        (_decimal(line.get("amount_brl")) or Decimal("0"))
        for line in financial["refund_lines"]
    )
    refund_total = _decimal(financial["recommended_refund_brl"]) or Decimal("0")
    if line_total != refund_total:
        raise ValueError("verifier rejected inconsistent refund totals")
    if output["assessment"]["case_status"] == "no_action" and refund_total != 0:
        raise ValueError("verifier rejected refund for a no-action case")
    actions = output["resolution_actions"]
    if len(actions) != len(set(actions)):
        raise ValueError("verifier rejected duplicate resolution actions")
    issue = output["assessment"]["primary_issue"]
    expected_status, expected_party, expected_action = ISSUE_INVARIANTS[issue]
    if output["assessment"]["case_status"] != expected_status:
        raise ValueError("verifier rejected case status inconsistent with primary issue")
    if actions != [expected_action]:
        raise ValueError("verifier rejected action inconsistent with primary issue")
    party_types = {
        party["party_type"]
        for party in output["root_cause_analysis"]["responsible_parties"]
    }
    if party_types != {expected_party}:
        raise ValueError("verifier rejected responsible party inconsistent with primary issue")
    if expected_party == "seller":
        seller_ids = set(output["affected_entities"]["seller_ids"])
        responsible_seller_ids = {
            party["party_id"]
            for party in output["root_cause_analysis"]["responsible_parties"]
            if party["party_type"] == "seller"
        }
        if not responsible_seller_ids or not responsible_seller_ids.issubset(seller_ids):
            raise ValueError("verifier rejected seller responsibility without seller evidence")
    if expected_status in {"no_action", "needs_investigation"}:
        if refund_total != 0 or financial["refund_lines"]:
            raise ValueError("verifier rejected refund for a non-resolution status")
    elif refund_total <= 0 or not financial["refund_lines"]:
        raise ValueError("verifier rejected action-required case without a positive refund")
    confidence = output["assessment"]["confidence"]
    if issue == "insufficient_evidence" and confidence > 0.55:
        raise ValueError("verifier rejected overconfident insufficient-evidence assessment")
    if output["data_conflicts"] and confidence > 0.95:
        raise ValueError("verifier rejected overconfidence in the presence of data conflicts")
    for conflict in output["data_conflicts"]:
        selected_source = conflict["selected_source"]
        if selected_source is not None and selected_source not in conflict["sources"]:
            raise ValueError("verifier rejected a conflict resolution with an unknown source")
    ranks = [cause["rank"] for cause in output["root_cause_analysis"]["ranked_causes"]]
    if ranks != list(range(1, len(ranks) + 1)):
        raise ValueError("verifier rejected non-contiguous root-cause ranks")
    trace.contracts.validate_output(output, f"verified output for {case_id}")


async def solve_case(
    case: dict[str, Any],
    gateway: EvidenceGateway,
    trace: TraceWriter,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Coordinate scoped specialists, policy evaluation, and deterministic verification.

    When *llm* is provided (Qwen 3 8B via OpenRouter), each specialist and the
    coordinator use the model as an advisory reasoning layer.  The deterministic
    rules remain authoritative so verification invariants are never violated.
    """
    case_id = case.get("case_id")
    request = case.get("customer_request")
    policy_version = case.get("policy_version")
    if not isinstance(case_id, str) or not isinstance(request, dict):
        raise ValueError("case is missing case_id or customer_request")
    order_id = request.get("claimed_order_id")
    claims = request.get("claims")
    if (
        not isinstance(order_id, str)
        or not isinstance(policy_version, str)
        or not isinstance(claims, list)
        or not claims
        or not isinstance(claims[0], dict)
        or not isinstance(claims[0].get("topic"), str)
    ):
        raise ValueError(f"{case_id}: invalid routing fields")

    primary_topic = claims[0]["topic"]
    plan = _tool_plan(primary_topic, order_id, policy_version)
    available_tools = set(await gateway.list_tools())
    policy_requests = plan.pop(POLICY_AGENT)
    domain_reports = await asyncio.gather(
        *(
            _run_specialist(
                actor=actor,
                requests=requests,
                case_id=case_id,
                available_tools=available_tools,
                gateway=gateway,
                trace=trace,
                llm=llm,
            )
            for actor, requests in plan.items()
        )
    )
    policy_report = await _run_specialist(
        actor=POLICY_AGENT,
        requests=policy_requests,
        case_id=case_id,
        available_tools=available_tools,
        gateway=gateway,
        trace=trace,
        llm=llm,
    )
    reports = [*domain_reports, policy_report]
    evidence_by_tool = {
        tool_name: evidence
        for report in reports
        for tool_name, evidence in report.evidence.items()
    }
    errors = {
        tool_name: error
        for report in reports
        for tool_name, error in report.errors.items()
    }

    # --- Coordinator LLM synthesis (advisory) ---
    if llm is not None and evidence_by_tool:
        try:
            specialist_summaries = {
                report.actor: {
                    "evidence_tools": list(report.evidence.keys()),
                    "error_tools": list(report.errors.keys()),
                }
                for report in reports
            }
            coordinator_analysis = await synthesize_assessment(
                llm,
                case=case,
                specialist_analyses=specialist_summaries,
            )
            logger.info("Coordinator LLM synthesis for %s: %s", case_id, coordinator_analysis)
        except Exception as exc:
            logger.warning("Coordinator LLM synthesis skipped for %s: %s", case_id, exc)

    collected_refs = {evidence["evidence_ref"] for evidence in evidence_by_tool.values()}
    coordinator_attrs: dict[str, Any] = {
        "specialist_count": len(reports),
        "error_count": len(errors),
    }
    if llm is not None:
        coordinator_attrs["llm_model"] = "qwen/qwen3-8b"

    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor="coordinator",
        target=VERIFIER_AGENT,
        decision_code="VERIFY_SYNTHESIZED_OUTPUT",
        evidence_refs=sorted(collected_refs),
        attributes=coordinator_attrs,
    )
    output = _build_output(case, evidence_by_tool, errors)
    _verify_output(
        output,
        case_id=case_id,
        collected_refs=collected_refs,
        trace=trace,
    )
    trace.emit(
        case_id=case_id,
        event_type="verification_completed",
        actor=VERIFIER_AGENT,
        target="coordinator",
        decision_code="OUTPUT_VALID",
        evidence_refs=output["evidence_refs"],
        attributes={"primary_issue": output["assessment"]["primary_issue"]},
    )
    return output

