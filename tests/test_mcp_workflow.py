from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from student_agent.contracts import Contracts
from student_agent.mcp_gateway import EvidenceGateway
from student_agent.workflow import _calibrate_confidence, _verify_output, solve_case

ROOT = Path(__file__).resolve().parents[1]


class ContractSpy:
    def __init__(self) -> None:
        self.validated: list[dict[str, Any]] = []

    def validate_evidence(self, evidence: dict[str, Any], _label: str) -> None:
        self.validated.append(evidence)


class SessionSpy:
    def __init__(self, evidence: dict[str, Any], *, is_error: bool = False) -> None:
        self.evidence = evidence
        self.is_error = is_error
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=[SimpleNamespace(name="get_order")])

    async def call_tool(self, tool_name: str, *, arguments: dict[str, str]) -> Any:
        self.calls.append((tool_name, arguments))
        return SimpleNamespace(
            is_error=self.is_error,
            structured_content=self.evidence,
            content=[SimpleNamespace(text="remote failure" if self.is_error else "")],
        )


def evidence(ref: str, domain: str, data: Any) -> dict[str, Any]:
    return {
        "schema_version": "day09-mcp-evidence-v1",
        "evidence_ref": ref,
        "result_hash": "sha256:" + "a" * 64,
        "domain": domain,
        "data": data,
        "warnings": [],
    }


def test_gateway_preserves_case_id_and_server_evidence_ref() -> None:
    source = evidence("ev_12345678901234567890", "order", {"order_id": "order-1"})
    session = SessionSpy(source)
    contracts = ContractSpy()
    gateway = EvidenceGateway(session, contracts)  # type: ignore[arg-type]

    result = asyncio.run(gateway.call("get_order", case_id="CASE_001", order_id="order-1"))

    assert session.calls == [
        ("get_order", {"case_id": "CASE_001", "order_id": "order-1"})
    ]
    assert result["evidence_ref"] == source["evidence_ref"]
    assert contracts.validated == [source]


def test_gateway_rejects_remote_tool_error() -> None:
    source = evidence("ev_12345678901234567890", "order", {})
    gateway = EvidenceGateway(SessionSpy(source, is_error=True), ContractSpy())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="get_order failed"):
        asyncio.run(gateway.call("get_order", case_id="CASE_001", order_id="order-1"))


class GatewaySpy:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    async def list_tools(self) -> list[str]:
        return sorted(self.responses)

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        self.calls.append((tool_name, case_id, arguments))
        return self.responses[tool_name]


class TraceSpy:
    def __init__(self) -> None:
        self.contracts = Contracts(ROOT / "contracts" / "schemas")
        self.events: list[dict[str, Any]] = []

    def emit(self, **event: Any) -> dict[str, Any]:
        self.events.append(event)
        return event


def test_specialists_consume_server_refs_within_case_scope() -> None:
    refs = {
        "get_order": "ev_order_12345678901234567890",
        "get_order_payments": "ev_payment_12345678901234567",
        "get_policy": "ev_policy_1234567890123456789",
    }
    responses = {
        "get_order": evidence(
            refs["get_order"],
            "order",
            {
                "order_id": "order-1",
                "order_status": "canceled",
                "order_purchase_timestamp": "2018-01-01T09:00:00-03:00",
            },
        ),
        "get_order_payments": evidence(
            refs["get_order_payments"],
            "payment",
            [{"order_id": "order-1", "payment_value": "79.00"}],
        ),
        "get_policy": evidence(
            refs["get_policy"],
            "policy",
            {
                "rules": {
                    "canceled_order_paid": {
                        "case_status": "action_required",
                        "recommended_action": "issue_refund",
                        "refund_brl": 79.0,
                        "responsible_parties": [
                            {"party_type": "platform", "party_id": None}
                        ],
                    }
                }
            },
        ),
    }
    gateway = GatewaySpy(responses)
    trace = TraceSpy()
    case = {
        "case_id": "CASE_001",
        "policy_version": "EC_POLICY_V1",
        "customer_request": {
            "claimed_order_id": "order-1",
            "claims": [
                {"claim_id": "claim-1", "topic": "canceled_order_paid"},
                {"claim_id": "claim-2", "topic": "requested_full_refund"},
            ],
        },
    }

    output = asyncio.run(solve_case(case, gateway, trace))  # type: ignore[arg-type]

    assert all(case_id == "CASE_001" for _, case_id, _ in gateway.calls)
    assert set(output["evidence_refs"]) == set(refs.values())
    consumed = {
        event["evidence_refs"][0]
        for event in trace.events
        if event["event_type"] == "tool_result_consumed"
    }
    assert consumed == set(refs.values())
    assert all(
        set(claim["evidence_refs"]).issubset(consumed)
        for claim in output["claim_assessments"]
    )
    assert output["assessment"] == {
        "primary_issue": "canceled_order_paid",
        "case_status": "action_required",
        "confidence": 0.96,
    }
    event_types = [event["event_type"] for event in trace.events]
    assert event_types.index("task_assigned") < event_types.index("tool_result_consumed")
    assert event_types.index("tool_result_consumed") < event_types.index("handoff")
    assert event_types.index("handoff") < event_types.index("policy_decided")
    assert event_types.index("policy_decided") < event_types.index("verification_completed")

    invalid = deepcopy(output)
    invalid["root_cause_analysis"]["responsible_parties"] = [
        {"party_type": "logistics_provider", "party_id": None}
    ]
    with pytest.raises(ValueError, match="responsible party"):
        _verify_output(
            invalid,
            case_id="CASE_001",
            collected_refs=set(refs.values()),
            trace=trace,  # type: ignore[arg-type]
        )


def test_conflicts_and_warnings_reduce_confidence() -> None:
    clean = {
        "get_order": evidence("ev_order_12345678901234567890", "order", {}),
        "get_policy": evidence("ev_policy_1234567890123456789", "policy", {}),
    }
    clean_confidence = _calibrate_confidence("duplicate_charge", clean, {}, [])
    clean["get_order"]["warnings"] = ["partial source"]
    conflicts = [
        {
            "field": "payment_timeline.events",
            "sources": ["purchase_window", "out_of_scope_events"],
            "selected_source": "purchase_window",
            "resolution_code": "CASE_TIME_SCOPE",
        }
    ]
    reduced_confidence = _calibrate_confidence(
        "duplicate_charge", clean, {}, conflicts
    )

    assert clean_confidence == 0.96
    assert reduced_confidence == 0.90
