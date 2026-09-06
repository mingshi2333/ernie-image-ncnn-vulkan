#!/usr/bin/env python3
"""Fail-closed protocol and aggregation for the frozen port benchmark."""
from __future__ import annotations
import hashlib, json, math, statistics
from pathlib import Path

MEASURED_PAIR_INDICES=tuple(range(5))
STAGES={"text","dit","scheduler","vae"}
SUCCESS={"ok","passed","success"}

def _nonempty(value): return isinstance(value,str) and bool(value.strip())
def _canonical(value): return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def _digest(value): return hashlib.sha256(value).hexdigest()

def _side(record,name):
    side=record.get(name)
    if not isinstance(side,dict): raise ValueError(f"{name} must contain its own actual measurement evidence")
    return side

def _stage_map(side,key):
    value=side.get(key)
    if not isinstance(value,dict) or set(value)!=STAGES or any(not _nonempty(v) for v in value.values()):
        raise ValueError(f"{key} must name actual text/dit/scheduler/vae values")
    return value

def validate_measurement(record:dict,frozen_case:dict|None=None)->None:
    if not isinstance(record,dict): raise ValueError("measurement must be an object")
    if record.get("phase")!="measured": raise ValueError("warmup or missing phase cannot enter formal aggregation")
    if not _nonempty(record.get("case_id")): raise ValueError("missing case_id")
    if record.get("pair_index") not in MEASURED_PAIR_INDICES: raise ValueError("pair_index must be one of five frozen repeats")
    expected_order="AB" if record["pair_index"]%2==0 else "BA"
    if record.get("order")!=expected_order: raise ValueError("record does not follow frozen AB/BA order")
    if record.get("trace") is not False: raise ValueError("formal measurement requires trace=false")
    sides=[_side(record,n) for n in ("candidate","reference")]
    for name,side in zip(("candidate","reference"),sides):
        if side.get("status") not in SUCCESS: raise ValueError(f"{name} status is not successful: {side.get('status')!r}")
        if side.get("quality_status") not in SUCCESS: raise ValueError(f"{name} quality status is not successful: {side.get('quality_status')!r}")
        if side.get("scope")!="end_to_end": raise ValueError(f"{name} timing scope must be end_to_end")
        if side.get("trace") is not False: raise ValueError(f"{name} trace must be false")
        for key in ("input_id","noise_sha256","model_id","weights_canonical_sha256","pe_identity"):
            if not _nonempty(side.get(key)): raise ValueError(f"{name} missing actual {key}")
        if side.get("noise_dtype")!="<f4": raise ValueError(f"{name} noise must be saved little-endian FP32 bytes")
        if side.get("weight_identity_status")!="proven": raise ValueError(f"{name} weight identity is not proven")
        _stage_map(side,"precision_by_stage"); _stage_map(side,"device_by_stage")
        started,finished=side.get("started_monotonic_ns"),side.get("finished_monotonic_ns")
        if not isinstance(started,int) or not isinstance(finished,int) or finished<=started:
            raise ValueError(f"{name} requires complete monotonic start/finish clocks")
        seconds=side.get("wall_seconds")
        if not isinstance(seconds,(int,float)) or isinstance(seconds,bool) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError(f"{name} has no positive complete wall time")
        if not math.isclose(float(seconds),(finished-started)/1e9,rel_tol=1e-6,abs_tol=1e-9):
            raise ValueError(f"{name} wall time does not match monotonic clocks")
    for key in ("input_id","noise_sha256","noise_dtype","model_id","weights_canonical_sha256","pe_identity","precision_by_stage"):
        if sides[0][key]!=sides[1][key]: raise ValueError(f"candidate/reference actual {key} mismatch")
    # Actual device maps are mandatory but may differ: the master protocol permits different scheduling.

    if frozen_case is not None:
        if record["case_id"]!=frozen_case.get("id") or frozen_case.get("split")!="performance":
            raise ValueError("record is not bound to its frozen performance case")
        expected={
            "input_id":frozen_case.get("prompt_sha256"), "noise_sha256":frozen_case.get("noise_sha256"),
            "noise_dtype":frozen_case.get("noise_dtype"), "model_id":_canonical(frozen_case.get("model_identity")).decode(),
            "shape":frozen_case.get("shape"), "shape_order":frozen_case.get("shape_order"),
            "steps":frozen_case.get("steps"), "cfg":frozen_case.get("cfg"), "pe":frozen_case.get("pe"),
            "precision_by_stage":frozen_case.get("dtype_by_stage"),
            "case_identity_sha256":_digest(_canonical(frozen_case)),
        }
        if frozen_case.get("mode")=="img2img":
            expected.update(input_image_sha256=frozen_case.get("input_image_sha256"),
                            decoded_rgb_sha256=frozen_case.get("decoded_rgb_sha256"),
                            strength=frozen_case.get("strength"),resize_policy=frozen_case.get("resize_policy"))
        for name,side in zip(("candidate","reference"),sides):
            for key,value in expected.items():
                if value is None or side.get(key)!=value:
                    raise ValueError(f"{name} actual {key} does not match frozen case")

def _frozen_performance(manifest,protocol):
    if not isinstance(manifest,dict) or manifest.get("schema_version")!=1: raise ValueError("trusted frozen manifest required")
    unsigned=dict(manifest); expected_hash=unsigned.pop("manifest_sha256",None)
    if _digest(_canonical(unsigned))!=expected_hash: raise ValueError("frozen manifest metadata checksum mismatch")
    protocol_entry=next((f for f in manifest.get("files",[]) if f.get("path")=="protocol.json"),None)
    if not protocol_entry or _digest(_canonical(protocol)+b'\n')!=protocol_entry.get("sha256"):
        raise ValueError("protocol does not match frozen manifest")
    perf=protocol.get("performance") if isinstance(protocol,dict) else None
    if (protocol.get("status")!="frozen_inputs_no_results" or protocol.get("performance_cases")!=6 or
        not isinstance(perf,dict) or perf.get("measured_pairs")!=5 or perf.get("warmups_per_port")!=1 or
        perf.get("trace") is not False or perf.get("order")!=["AB","BA","AB","BA","AB"]):
        raise ValueError("frozen performance protocol mismatch")
    cases=[c for c in manifest.get("cases",[]) if c.get("split")=="performance"]
    if len(cases)!=6 or len({c.get("id") for c in cases})!=6: raise ValueError("frozen performance cases mismatch")
    return {c["id"]:c for c in cases}

def summarize_pairs(pairs:list[dict],manifest:dict,protocol:dict)->dict:
    frozen=_frozen_performance(manifest,protocol)
    expected={(case,i) for case in frozen for i in MEASURED_PAIR_INDICES}
    seen={}; invalid=[]; duplicate=[]; unexpected=[]
    for position,pair in enumerate(pairs):
        key=(pair.get("case_id"),pair.get("pair_index")) if isinstance(pair,dict) else (None,None)
        if key not in expected: unexpected.append({"position":position,"case_id":key[0],"pair_index":key[1]})
        elif key in seen: duplicate.append({"case_id":key[0],"pair_index":key[1]})
        else: seen[key]=pair
        try: validate_measurement(pair,frozen.get(key[0]))
        except (TypeError,ValueError) as error:
            invalid.append({"position":position,"case_id":key[0],"pair_index":key[1],"reason":str(error)})
    missing=sorted(expected-set(seen))
    complete=not missing and not duplicate and not unexpected and not invalid and len(pairs)==30
    cases=[]; ratios=[]
    if complete:
        for case in frozen:
            candidate=statistics.median(_side(seen[(case,i)],"candidate")["wall_seconds"] for i in MEASURED_PAIR_INDICES)
            reference=statistics.median(_side(seen[(case,i)],"reference")["wall_seconds"] for i in MEASURED_PAIR_INDICES)
            ratio=reference/candidate; ratios.append(ratio)
            cases.append({"case_id":case,"pair_count":5,"candidate_median_seconds":candidate,
                          "reference_median_seconds":reference,"reference_over_candidate_ratio":ratio})
    unavailable=sorted({x[0] for x in missing}|{x["case_id"] for x in invalid if x["case_id"]})
    return {"status":"complete" if complete else "incomplete","expected_pair_count":30,
            "valid_pair_count":30 if complete else 0,
            "candidate_median_seconds":statistics.median(c["candidate_median_seconds"] for c in cases) if complete else None,
            "reference_median_seconds":statistics.median(c["reference_median_seconds"] for c in cases) if complete else None,
            "geomean_ratio":math.exp(sum(math.log(r) for r in ratios)/6) if complete else None,
            "cases":cases,"unavailable_case_ids":unavailable,"missing_records":[list(x) for x in missing],
            "duplicate_records":duplicate,"unexpected_records":unexpected,"invalid_pairs":invalid}

def paired_schedule(case_id:str,measured_pairs:int=5)->list[dict]:
    if not _nonempty(case_id) or measured_pairs!=5:
        raise ValueError("formal schedule requires a case ID and five pairs")
    runs=[{"case_id":case_id,"phase":"warmup","port":"candidate","new_process":True},
          {"case_id":case_id,"phase":"warmup","port":"reference","new_process":True}]
    for index in MEASURED_PAIR_INDICES:
        order=("candidate","reference") if index%2==0 else ("reference","candidate")
        for sequence,port in enumerate(order):
            runs.append({"case_id":case_id,"phase":"measured","pair_index":index,"sequence":sequence,
                         "order":"AB" if index%2==0 else "BA","port":port,"new_process":True})
    return runs

def build_protocol(cases:list[dict],calibration:dict|None,available_disk_bytes:int|None,measured_pairs:int=5)->dict:
    ids=[c.get("id") or c.get("case_id") for c in cases]; reasons=[]
    if len(ids)!=6 or len(set(ids))!=6 or any(not _nonempty(x) for x in ids): reasons.append("six unique frozen performance cases required")
    if measured_pairs!=5: reasons.append("formal measured pair count is frozen at five")
    required=("seconds_512","seconds_1024","seconds_long_text","seconds_pe","trace_bytes_per_case")
    if not calibration or any(not isinstance(calibration.get(k),(int,float)) or isinstance(calibration.get(k),bool)
                              or not math.isfinite(calibration[k]) or calibration[k]<=0 for k in required):
        reasons.append("positive finite empirical timing and trace-size calibration is required")
    if not isinstance(available_disk_bytes,int) or available_disk_bytes<=0: reasons.append("positive available disk measurement is required")
    estimated=required_disk=timeout=None
    if not reasons:
        slowest=max(float(calibration[k]) for k in required[:-1]); estimated=slowest*12*6
        timeout=math.ceil(slowest*1.5+60); required_disk=math.ceil(float(calibration["trace_bytes_per_case"])*6*1.2)
        if available_disk_bytes<required_disk: reasons.append("insufficient disk for full formal archive")
    return {"schema_version":1,"status":"incomplete" if reasons else "ready","incomplete_reasons":reasons,
            "formal_case_ids":ids,"formal_denominator":30,"warmups_per_port":1,"measured_pairs":5,
            "process_scope":"new_process","clock":"host_monotonic_ns_process_launch_through_output_close","trace":False,
            "schedule":[r for case in ids if _nonempty(case) for r in paired_schedule(case)],
            "budget":{"formal_process_runs":72,"estimated_seconds":estimated,"estimated_hours":estimated/3600 if estimated else None,
                      "available_disk_bytes":available_disk_bytes,"required_disk_bytes":required_disk,"single_case_timeout_seconds":timeout},
            "archive":{"trace_validation":"validate each case before packaging","sha256_required":True,
                       "disk_shortage_policy":"shrink development diagnostics only; preserve formal denominator and failures"}}

def serialize_protocol(path:Path,protocol:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+"\n")
