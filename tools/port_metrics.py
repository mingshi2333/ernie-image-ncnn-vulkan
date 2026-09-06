#!/usr/bin/env python3
"""Fail-closed protocol and aggregation for paired end-to-end port timings."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


REQUIRED_IDENTITY = ("input_id", "noise_id", "model_id", "precision", "pe_enabled")
SUCCESS = {"ok", "passed", "success"}


def _side(record: dict, name: str) -> dict:
    nested = record.get(name)
    if isinstance(nested, dict):
        return nested
    return {
        "status": record.get(f"{name}_status"),
        "quality_status": record.get(f"{name}_quality_status"),
        "wall_seconds": record.get(f"{name}_seconds"),
        "started_monotonic_ns": record.get(f"{name}_started_monotonic_ns"),
        "finished_monotonic_ns": record.get(f"{name}_finished_monotonic_ns"),
        "scope": record.get(f"{name}_scope", record.get("scope")),
    }


def validate_measurement(record: dict) -> None:
    """Reject records that cannot support an end-to-end speed comparison."""
    if not isinstance(record, dict):
        raise ValueError("measurement must be an object")
    if not record.get("case_id"):
        raise ValueError("missing case_id")
    for key in REQUIRED_IDENTITY:
        if key not in record or record[key] is None:
            raise ValueError(f"missing {key}")
    if not isinstance(record["pe_enabled"], bool):
        raise ValueError("pe_enabled must be boolean")
    if record.get("trace") is not False:
        raise ValueError("formal measurement requires trace=false")

    for name in ("candidate", "reference"):
        side = _side(record, name)
        status = side.get("status")
        if status not in SUCCESS:
            raise ValueError(f"{name} status is not successful: {status!r}")
        if side.get("quality_status") not in SUCCESS:
            raise ValueError(f"{name} quality status is not successful: {side.get('quality_status')!r}")
        started = side.get("started_monotonic_ns")
        finished = side.get("finished_monotonic_ns")
        if not isinstance(started, int) or not isinstance(finished, int) or finished <= started:
            raise ValueError(f"{name} requires complete monotonic start/finish clocks")
        seconds = side.get("wall_seconds")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"{name} has no positive complete wall time")
        scope = side.get("scope", record.get("scope"))
        if scope != "end_to_end":
            raise ValueError(f"{name} timing scope must be end_to_end")
        elapsed = (finished - started) / 1_000_000_000
        if not math.isclose(float(seconds), elapsed, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"{name} wall time does not match monotonic clocks")
        for key in REQUIRED_IDENTITY:
            if key in side and side[key] != record[key]:
                raise ValueError(f"{name} {key} does not match pair")


def summarize_pairs(pairs: list[dict]) -> dict:
    """Aggregate only a complete valid denominator; never score survivors."""
    unavailable: list[str] = []
    invalid: list[dict] = []
    by_case: dict[str, dict[str, list[float]]] = {}

    for index, pair in enumerate(pairs):
        case_id = pair.get("case_id", f"index-{index}") if isinstance(pair, dict) else f"index-{index}"
        try:
            validate_measurement(pair)
        except (TypeError, ValueError) as error:
            unavailable.append(case_id)
            invalid.append({"case_id": case_id, "reason": str(error)})
            continue
        candidate = float(_side(pair, "candidate")["wall_seconds"])
        reference = float(_side(pair, "reference")["wall_seconds"])
        bucket = by_case.setdefault(case_id, {"candidate": [], "reference": []})
        bucket["candidate"].append(candidate)
        bucket["reference"].append(reference)

    complete = bool(pairs) and not unavailable
    cases = []
    ratios = []
    for case_id, values in by_case.items():
        candidate = statistics.median(values["candidate"])
        reference = statistics.median(values["reference"])
        cases.append({"case_id": case_id, "pair_count": len(values["candidate"]),
                      "candidate_median_seconds": candidate,
                      "reference_median_seconds": reference,
                      "reference_over_candidate_ratio": reference / candidate})
        ratios.append(reference / candidate)
    return {
        "status": "complete" if complete else "incomplete",
        "expected_pair_count": len(pairs),
        "valid_pair_count": sum(len(value["candidate"]) for value in by_case.values()),
        "candidate_median_seconds": statistics.median([c["candidate_median_seconds"] for c in cases]) if complete else None,
        "reference_median_seconds": statistics.median([c["reference_median_seconds"] for c in cases]) if complete else None,
        "geomean_ratio": math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if complete else None,
        "cases": cases if complete else [],
        "unavailable_case_ids": unavailable,
        "invalid_pairs": invalid,
    }


def paired_schedule(case_id: str, measured_pairs: int = 5) -> list[dict]:
    """One warmup per port followed by alternating AB/BA new-process pairs."""
    if measured_pairs < 1:
        raise ValueError("measured_pairs must be positive")
    runs = [
        {"case_id": case_id, "phase": "warmup", "port": "candidate", "new_process": True},
        {"case_id": case_id, "phase": "warmup", "port": "reference", "new_process": True},
    ]
    for pair_index in range(measured_pairs):
        order = ("candidate", "reference") if pair_index % 2 == 0 else ("reference", "candidate")
        for sequence, port in enumerate(order):
            runs.append({"case_id": case_id, "phase": "measured", "pair_index": pair_index,
                         "sequence": sequence, "order": "AB" if pair_index % 2 == 0 else "BA",
                         "port": port, "new_process": True})
    return runs


def build_protocol(cases: list[dict], calibration: dict | None, available_disk_bytes: int | None,
                   measured_pairs: int = 5) -> dict:
    """Build an honest protocol/budget object; absent calibration stays incomplete."""
    case_ids = [case.get("id") or case.get("case_id") for case in cases]
    reasons = []
    if not cases or any(not case_id for case_id in case_ids):
        reasons.append("frozen case IDs are required")
    required = ("seconds_512", "seconds_1024", "seconds_long_text", "seconds_pe", "trace_bytes_per_case")
    if not calibration or any(not isinstance(calibration.get(k), (int, float)) or calibration[k] < 0 for k in required):
        reasons.append("empirical 512/1024/long-text/PE timing and trace-size calibration is required")
    estimated_seconds = None
    required_disk = None
    timeout = None
    if not reasons:
        slowest = max(float(calibration[k]) for k in required[:-1])
        processes_per_case = 2 + 2 * measured_pairs
        estimated_seconds = slowest * processes_per_case * len(cases)
        timeout = math.ceil(slowest * 1.5 + 60)
        required_disk = math.ceil(float(calibration["trace_bytes_per_case"]) * len(cases) * 1.2)
        if available_disk_bytes is None:
            reasons.append("available disk measurement is required")
        elif available_disk_bytes < required_disk:
            reasons.append("insufficient disk for the formal full-denominator archive")
    return {
        "schema_version": 1,
        "status": "incomplete" if reasons else "ready",
        "incomplete_reasons": reasons,
        "formal_case_ids": case_ids,
        "formal_denominator": len(case_ids),
        "warmups_per_port": 1,
        "measured_pairs": measured_pairs,
        "process_scope": "new_process",
        "clock": "host_monotonic_ns_process_launch_through_output_close",
        "trace": False,
        "schedule": [run for case_id in case_ids if case_id for run in paired_schedule(case_id, measured_pairs)],
        "budget": {"formal_process_runs": len(case_ids) * (2 + 2 * measured_pairs),
                   "estimated_seconds": estimated_seconds,
                   "estimated_hours": estimated_seconds / 3600 if estimated_seconds is not None else None,
                   "available_disk_bytes": available_disk_bytes,
                   "required_disk_bytes": required_disk,
                   "single_case_timeout_seconds": timeout},
        "archive": {"trace_validation": "validate each case before packaging",
                    "sha256_required": True,
                    "disk_shortage_policy": "shrink development diagnostics only; preserve formal denominator and failures"},
    }


def serialize_protocol(path: Path, protocol: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
