#!/usr/bin/env python3
"""Run every example utterance in a Will-It-Jev taskonomy through the live router.

The router itself writes route capture records to logs/routing_capture.jsonl and
inserts them into ClickHouse. This script does not write duplicate ClickHouse
records; it correlates each request with the router's captured RouteRecord and
writes a timestamped JSON report, updating it after every example.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def request_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None,
                 timeout: float = 10.0) -> tuple[int, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, {"raw_text": raw}


def load_cases(taxonomy_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    categories = taxonomy.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{taxonomy_path} must contain a non-empty 'categories' array")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for category in categories:
        category_id = category.get("id")
        if isinstance(category_id, bool) or not isinstance(category_id, int) or category_id < 1:
            raise ValueError(f"Every category must have a positive integer 'id', got {category_id!r}")
        if category_id in seen_ids:
            raise ValueError(f"Duplicate category id: {category_id}")
        seen_ids.add(category_id)
        examples = category.get("examples", [])
        if not isinstance(examples, list):
            raise ValueError(f"categories[{category_id}].examples must be an array")
        for index, example in enumerate(examples, start=1):
            if not isinstance(example, str) or not example.strip():
                raise ValueError(f"Empty/non-string example in category {category_id}")
            cases.append({
                "case_id": f"{category_id}:{index:02d}",
                "expected_task_category": category_id,
                "category_slug": category.get("name"),
                "category_name": category.get("desc", category.get("name", str(category_id))),
                "example_index": index,
                "prompt": example,
            })
    if not cases:
        raise ValueError("No category examples found in taxonomy")
    return taxonomy, cases


def read_complete_jsonl(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read complete appended JSONL records from offset; leave a partial line unread."""
    if not path.exists():
        return [], offset
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read()
    last_newline = data.rfind(b"\n")
    if last_newline < 0:
        return [], offset
    complete = data[: last_newline + 1]
    new_offset = offset + last_newline + 1
    records: list[dict[str, Any]] = []
    for line in complete.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records, new_offset


def wait_for_capture(path: Path, offset: int, prompt: str, timeout: float,
                     poll_interval: float = 0.10) -> tuple[dict[str, Any] | None, int, list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout
    seen: list[dict[str, Any]] = []
    current_offset = offset
    while time.monotonic() < deadline:
        records, current_offset = read_complete_jsonl(path, current_offset)
        seen.extend(records)
        for record in records:
            if record.get("prompt_raw") == prompt:
                return record, current_offset, seen
        time.sleep(poll_interval)
    return None, current_offset, seen


def summarize(samples: list[dict[str, Any]], categories: list[dict[str, Any]]) -> dict[str, Any]:
    backend_counts: Counter[str] = Counter()
    per_category: dict[str, dict[str, Any]] = {}
    for category in categories:
        cid = category["id"]
        rows = [s for s in samples if s["expected_task_category"] == cid]
        per_category[str(cid)] = {
            "slug": category.get("name"),
            "name": category.get("desc", category.get("name", str(cid))),
            "examples": len(rows),
            "route_records_found": sum(s.get("route_record") is not None for s in rows),
            "http_2xx": sum(isinstance(s.get("http_status"), int) and 200 <= s["http_status"] < 300 for s in rows),
            "final_backend_counts": dict(Counter(
                s["route_record"].get("final_backend", "unknown")
                if s.get("route_record") else "no_route_record"
                for s in rows
            )),
            "failovers": sum(bool((s.get("route_record") or {}).get("is_failover")) for s in rows),
        }
        for s in rows:
            record = s.get("route_record")
            backend = record.get("final_backend", "unknown") if record else "no_route_record"
            backend_counts[backend] += 1

    total = len(samples)
    found = sum(s.get("route_record") is not None for s in samples)
    final_jev = backend_counts.get("jev", 0)
    final_llm = backend_counts.get("llm", 0)
    return {
        "samples_total": total,
        "samples_completed": len(samples),
        "route_records_found": found,
        "http_2xx": sum(isinstance(s.get("http_status"), int) and 200 <= s["http_status"] < 300 for s in samples),
        "final_backend_counts": dict(backend_counts),
        "final_jev_share_of_all_samples": round(final_jev / total, 4) if total else None,
        "final_jev_share_of_recorded_routes": round(final_jev / found, 4) if found else None,
        "failovers": sum(bool((s.get("route_record") or {}).get("is_failover")) for s in samples),
        "by_task_category": per_category,
        "interpretation": (
            "These are observed routing/execution outcomes, not Jev-eligibility accuracy. "
            "Accuracy requires human-reviewed jev_eligible gold labels. The taxonomy's examples "
            "supply expected task categories, but the current router endpoint does not return a "
            "prediction among the 20 taskonomy leaf categories."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=Path("taskonomies/help_desk.json"))
    parser.add_argument("--router", default="http://127.0.0.1:8080", help="Will-It-Jev base URL")
    parser.add_argument("--capture-log", type=Path, default=Path("logs/routing_capture.jsonl"))
    parser.add_argument("--output", type=Path, help="Report JSON path (default: timestamped file in evals/results)")
    parser.add_argument("--request-timeout", type=float, default=180.0, help="Per request timeout in seconds")
    parser.add_argument("--capture-timeout", type=float, default=10.0, help="Seconds to wait for the matching JSONL route record")
    parser.add_argument("--limit", type=int, help="Only run the first N examples, useful for a smoke test")
    args = parser.parse_args()

    taxonomy_path = args.taxonomy.resolve()
    capture_path = args.capture_log.resolve()
    router = args.router.rstrip("/")
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("evals/results") / f"{taxonomy_path.stem}_{stamp}.json"
    output = output.resolve()

    try:
        taxonomy, all_cases = load_cases(taxonomy_path)
    except Exception as exc:
        print(f"ERROR: cannot load taskonomy: {exc}", file=sys.stderr)
        return 2
    cases = all_cases[: args.limit] if args.limit is not None else all_cases
    if not cases:
        print("ERROR: no cases selected", file=sys.stderr)
        return 2

    try:
        status, health = request_json(f"{router}/healthz", timeout=5)
        if not (200 <= status < 300):
            raise RuntimeError(f"health check returned HTTP {status}: {health}")
    except Exception as exc:
        print(f"ERROR: cannot reach Will-It-Jev at {router}: {exc}", file=sys.stderr)
        return 2

    # Capture only records generated after this evaluation begins.
    if capture_path.exists():
        capture_offset = capture_path.stat().st_size
    else:
        print(f"ERROR: route capture log not found: {capture_path}\nRun this from the repository root and confirm the router is running.", file=sys.stderr)
        return 2

    started_at = utc_now()
    report: dict[str, Any] = {
        "report_schema_version": "1.0.0",
        "run_id": f"{taxonomy.get('taxonomy_id', taxonomy_path.stem)}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}",
        "taxonomy_id": taxonomy.get("taxonomy_id", taxonomy_path.stem),
        "taxonomy_version": taxonomy.get("version"),
        "taxonomy_path": str(taxonomy_path),
        "router_base_url": router,
        "capture_log": str(capture_path),
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "notes": [
            "Each category example is sent as a single-turn prompt to POST /v1/completions.",
            "Will-It-Jev writes routing records to logs/routing_capture.jsonl and asynchronously inserts them into ClickHouse; this script does not duplicate those inserts.",
            "The router's post-route tagger currently uses its own coarse domain labels; this script does not claim to evaluate the 20 leaf taskonomy categories as predicted labels.",
        ],
        "samples": [],
        "summary": {},
    }
    atomic_write_json(output, report)
    print(f"Running {len(cases)} examples from {taxonomy_path}")
    print(f"Router: {router}")
    print(f"Report: {output}")

    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['expected_task_category']}: {case['prompt']}")
        started_case = utc_now()
        began = time.monotonic()
        request_error = None
        response_body: Any = None
        try:
            status, response_body = request_json(
                f"{router}/v1/completions",
                method="POST",
                payload={"prompt": case["prompt"]},
                timeout=args.request_timeout,
            )
        except Exception as exc:
            status = None
            request_error = str(exc)

        elapsed_ms = round((time.monotonic() - began) * 1000, 3)
        route_record, capture_offset, _seen = wait_for_capture(
            capture_path, capture_offset, case["prompt"], args.capture_timeout
        )

        route_summary = None
        if route_record is not None:
            final = route_record.get("final_routing") or {}
            route_summary = {
                "request_id": route_record.get("request_id"),
                "timestamp": route_record.get("timestamp"),
                "final_backend": final.get("backend"),
                "final_tier": final.get("tier"),
                "cascade_decision_tier": route_record.get("cascade_decision_tier"),
                "router_classification_latency_us": route_record.get("router_classification_latency_us"),
                "total_latency_us": final.get("latency_us"),
                "downstream_tier": route_record.get("downstream_tier"),
                "downstream_outcome": route_record.get("downstream_outcome"),
                "is_failover": route_record.get("is_failover", False),
                "failure_reason": route_record.get("failure_reason"),
                "tier1_outcome": route_record.get("tier1_outcome"),
                "tier2_outcome": route_record.get("tier2_outcome"),
                "tier3_outcome": route_record.get("tier3_outcome"),
                "tier4_outcome": route_record.get("tier4_outcome"),
            }

        sample_result = {
            **case,
            "started_at": started_case,
            "elapsed_ms": elapsed_ms,
            "http_status": status,
            "request_error": request_error,
            "response": response_body,
            "route_record": route_summary,
            "router_capture_seen": route_record is not None,
        }
        report["samples"].append(sample_result)
        report["summary"] = summarize(report["samples"], taxonomy["categories"])
        atomic_write_json(output, report)
        backend = (route_summary or {}).get("final_backend", "no-route-record")
        print(f"    HTTP={status} final_backend={backend} route_record={'yes' if route_record else 'NO'} elapsed={elapsed_ms:.0f}ms")

    report["finished_at"] = utc_now()
    report["status"] = "completed" if len(report["samples"]) == len(cases) else "incomplete"
    report["summary"] = summarize(report["samples"], taxonomy["categories"])
    report["summary"]["requested_examples"] = len(cases)
    report["summary"]["taxonomy_examples_available"] = len(all_cases)
    report["summary"]["clickhouse_delivery"] = (
        "The router attempts asynchronous ClickHouse insertion for each captured route. "
        "Check router logs for insertion failures; route capture presence alone does not prove ClickHouse acknowledged the insert."
    )
    atomic_write_json(output, report)
    print("\nSummary")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"\nSaved: {output}")
    if report["summary"]["route_records_found"] != len(cases):
        print("WARNING: one or more route capture records were not found. Check router logs and --capture-log.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
