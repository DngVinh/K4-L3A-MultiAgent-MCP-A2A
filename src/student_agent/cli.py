from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .cases import load_case_set
from .config import Settings, load_llm_client
from .contracts import Contracts
from .mcp_gateway import connect_gateway
from .submission import package_submission, validate_artifacts
from .trace import TraceWriter
from .workflow import solve_case

logger = logging.getLogger(__name__)


def _root(value: str) -> Path:
    return Path(value).resolve()


async def _show_tools(root: Path) -> None:
    settings = Settings.load(root)
    contracts = Contracts(root / "contracts" / "schemas")
    async with connect_gateway(settings.mcp_endpoint, settings.team_api_key, contracts) as gateway:
        for tool in await gateway.list_tools():
            print(tool)


async def _run(root: Path, resume: bool = False) -> None:
    settings = Settings.load(root)
    case_set = load_case_set(root)
    contracts = Contracts(root / "contracts" / "schemas")
    output_root = root / "outputs"
    trace_path = root / "traces" / "trace.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    finalized_cases: set[str] = set()
    if resume and trace_path.exists():
        existing_lines = [
            line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        events_by_case: dict[str, list[dict[str, Any]]] = {}
        for raw_line in existing_lines:
            try:
                ev = json.loads(raw_line)
                events_by_case.setdefault(ev.get("case_id", ""), []).append(ev)
            except Exception:
                pass

        valid_trace_lines: list[str] = []
        for case_id in case_set.case_ids:
            case_events = events_by_case.get(case_id, [])
            output_file = output_root / f"{case_id}.json"
            if (
                any(e.get("event_type") == "case_finalized" for e in case_events)
                and output_file.exists()
            ):
                try:
                    out_data = json.loads(output_file.read_text(encoding="utf-8"))
                    contracts.validate_output(out_data, f"outputs/{case_id}.json")
                    finalized_cases.add(case_id)
                    for raw_line in existing_lines:
                        try:
                            if json.loads(raw_line).get("case_id") == case_id:
                                valid_trace_lines.append(raw_line)
                        except Exception:
                            pass
                except Exception:
                    pass
            else:
                if output_file.exists():
                    output_file.unlink()

        trace_path.write_text(
            "\n".join(valid_trace_lines) + ("\n" if valid_trace_lines else ""), encoding="utf-8"
        )
        remaining = len(case_set.case_ids) - len(finalized_cases)
        print(f"Resuming: {len(finalized_cases)} cases already valid. Remaining: {remaining} cases.")
    else:
        for stale in output_root.glob("*.json"):
            stale.unlink()
        trace_path.unlink(missing_ok=True)

    trace = TraceWriter(trace_path, contracts)

    # --- Initialise Qwen 3 8B via OpenRouter (< 10B param requirement) ---
    llm = load_llm_client()
    if llm is not None:
        logger.info("LLM enabled: Qwen 3 8B via OpenRouter")
    else:
        logger.warning("OPENROUTER_API_KEY not set — running without LLM reasoning")

    total_cases = len(case_set.case_ids)
    print(f"Processing {total_cases} cases...")
    try:
        # Pre-check tools once
        async with connect_gateway(settings.mcp_endpoint, settings.team_api_key, contracts) as initial_gw:
            discovered_tools = await initial_gw.list_tools()
            if not discovered_tools:
                raise RuntimeError("MCP Gateway returned no tools")

        for index, case_id in enumerate(case_set.case_ids, 1):
            if case_id in finalized_cases:
                print(f"[{index:3d}/{total_cases}] {case_id}: already finalized - skipping")
                continue
            case = case_set.cases[case_id]
            trace.emit(case_id=case_id, event_type="case_received", actor="coordinator")
            async with connect_gateway(settings.mcp_endpoint, settings.team_api_key, contracts) as gateway:
                output = await solve_case(case, gateway, trace, llm=llm)
            contracts.validate_output(output, f"outputs/{case_id}.json")
            if output.get("case_id") != case_id:
                raise ValueError(f"solver returned a mismatched case_id for {case_id}")
            target = output_root / f"{case_id}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(target)
            trace.emit(case_id=case_id, event_type="case_finalized", actor="coordinator")
            issue = output["assessment"]["primary_issue"]
            status = output["assessment"]["case_status"]
            print(f"[{index:3d}/{total_cases}] {case_id}: {issue} ({status}) - OK")
        print(f"Completed all {total_cases} cases successfully.")
    finally:
        if llm is not None:
            await llm.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Day09 L3A student workflow")
    result.add_argument("--root", default=".", help="repository root (default: current directory)")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-inputs", help="validate case-set.json and all 100 inputs")
    commands.add_parser("mcp-tools", help="authenticate and list discovered MCP tools")
    run_cmd = commands.add_parser("run", help="run the implemented workflow for all cases")
    run_cmd.add_argument(
        "--resume",
        action="store_true",
        help="resume from where it left off, skipping already finalized cases",
    )
    commands.add_parser("validate", help="validate outputs and observable trace")
    package = commands.add_parser("package", help="validate and build the submission ZIP")
    package.add_argument("--output", default="dist/submission.zip")
    return result


def main() -> None:
    args = parser().parse_args()
    root = _root(args.root)
    try:
        if args.command == "validate-inputs":
            case_set = load_case_set(root)
            print(
                f"OK: {case_set.variant_id} / {case_set.version} / "
                f"{len(case_set.case_ids)} cases"
            )
        elif args.command == "mcp-tools":
            asyncio.run(_show_tools(root))
        elif args.command == "run":
            asyncio.run(_run(root, resume=args.resume))
        elif args.command == "validate":
            case_set = load_case_set(root)
            contracts = Contracts(root / "contracts" / "schemas")
            _, trace = validate_artifacts(root, case_set, contracts)
            print(f"OK: {len(case_set.case_ids)} outputs / {len(trace)} trace events")
        elif args.command == "package":
            destination = package_submission(root, root / args.output)
            print(f"OK: {destination}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
