import json, signal, subprocess, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"tools"))
from benchmark_pipeline import run_timed_command
from port_metrics import FORMAL_CASE_IDS, build_protocol, paired_schedule, summarize_pairs, validate_measurement

PRECISION={"text":"fp32","dit":"fp32","scheduler":"fp32","vae":"fp32"}
def side(seconds=2.0,device="vulkan"):
    start=1_000_000_000
    return {"status":"ok","quality_status":"passed","scope":"end_to_end","trace":False,
            "input_id":"a"*64,"noise_sha256":"b"*64,"noise_dtype":"<f4","model_id":"official-turbo",
            "weights_canonical_sha256":"c"*64,"weight_identity_status":"proven","pe_identity":"disabled",
            "precision_by_stage":dict(PRECISION),"device_by_stage":{k:device for k in PRECISION},
            "started_monotonic_ns":start,"finished_monotonic_ns":start+int(seconds*1e9),"wall_seconds":seconds}
def pair(case="performance-0",index=0):
    return {"case_id":case,"pair_index":index,"phase":"measured","order":"AB" if index%2==0 else "BA",
            "trace":False,"candidate":side(2.0,"vulkan"),"reference":side(4.0,"cpu")}
def full_pairs(): return [pair(case,index) for case in FORMAL_CASE_IDS for index in range(5)]

class PortMetricsTest(unittest.TestCase):
    def test_frozen_denominator_completes_and_allows_different_devices(self):
        result=summarize_pairs(full_pairs())
        self.assertEqual(result["status"],"complete");self.assertEqual(result["expected_pair_count"],30)
        self.assertEqual(result["geomean_ratio"],2.0);self.assertEqual(len(result["cases"]),6)

    def test_missing_repeat_duplicate_unexpected_and_warmup_are_incomplete(self):
        variants=[]
        missing=full_pairs()[:-1];variants.append(missing)
        duplicate=full_pairs();duplicate[-1]=duplicate[0];variants.append(duplicate)
        unexpected=full_pairs();unexpected[-1]=dict(unexpected[-1],case_id="other");variants.append(unexpected)
        warmup=full_pairs();warmup[-1]=dict(warmup[-1],phase="warmup");variants.append(warmup)
        for records in variants:
            with self.subTest():
                result=summarize_pairs(records);self.assertEqual(result["status"],"incomplete")
                self.assertIsNone(result["geomean_ratio"])

    def test_failed_reference_is_not_infinite_speedup(self):
        records=full_pairs();records[0]["reference"]["status"]="oom";records[0]["reference"]["quality_status"]="unknown"
        result=summarize_pairs(records)
        self.assertIsNone(result["geomean_ratio"]);self.assertIn("performance-0",result["unavailable_case_ids"])

    def test_side_identity_mismatch_is_rejected_for_that_reason(self):
        record=pair();record["candidate"]["noise_sha256"]="d"*64
        with self.assertRaisesRegex(ValueError,"noise_sha256 mismatch"):validate_measurement(record)

    def test_each_side_must_supply_actual_evidence_and_trace_false(self):
        mutations=[("input_id",None),("noise_dtype","float32"),("weights_canonical_sha256",None),
                   ("precision_by_stage",{"dit":"fp32"}),("device_by_stage",None),("trace",True)]
        for key,value in mutations:
            record=pair();record["candidate"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_measurement(record)

    def test_precision_and_pe_mismatch_rejected_but_device_mismatch_allowed(self):
        record=pair();validate_measurement(record)
        record=pair();record["reference"]["precision_by_stage"]["dit"]="fp16"
        with self.assertRaisesRegex(ValueError,"precision_by_stage mismatch"):validate_measurement(record)
        record=pair();record["reference"]["pe_identity"]="enabled:hash"
        with self.assertRaisesRegex(ValueError,"pe_identity mismatch"):validate_measurement(record)

    def test_schedule_is_fixed(self):
        schedule=paired_schedule("performance-0");self.assertEqual(len(schedule),12)
        self.assertEqual([r["order"] for r in schedule[2::2]],["AB","BA","AB","BA","AB"])
        with self.assertRaises(ValueError):paired_schedule("performance-0",4)

    def test_protocol_rejects_short_duplicate_or_zero_calibration(self):
        cases=[{"id":case} for case in FORMAL_CASE_IDS]
        calibration={"seconds_512":1,"seconds_1024":2,"seconds_long_text":3,"seconds_pe":4,"trace_bytes_per_case":5}
        self.assertEqual(build_protocol(cases,calibration,1000)["status"],"ready")
        for bad_cases,bad_calibration in ((cases[:1],calibration),(cases[:-1]+cases[:1],calibration),(cases,{**calibration,"seconds_pe":0})):
            self.assertEqual(build_protocol(bad_cases,bad_calibration,1000)["status"],"incomplete")

    def test_benchmark_reports_unsupported_img2img_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"out"
            run=subprocess.run([sys.executable,str(Path(__file__).parents[1]/"tools"/"benchmark_pipeline.py"),
                "--model",str(Path(temporary)/"model"),"--output",str(output),"--input-image","x"],capture_output=True)
            self.assertEqual(run.returncode,2);self.assertEqual(json.loads((output/"result.json").read_text())["status"],"incomplete")

    def test_synthetic_timeout_persists_timing_and_category(self):
        with tempfile.TemporaryDirectory() as temporary:
            result=run_timed_command([sys.executable,"-c","import time;time.sleep(2)"],0.02,Path(temporary)/"log")
        self.assertEqual(result["failure_category"],"timeout");self.assertTrue(result["timed_out"])
        self.assertGreater(result["wall_finished_monotonic_ns"],result["wall_started_monotonic_ns"])

    def test_synthetic_crash_persists_signal_and_category(self):
        with tempfile.TemporaryDirectory() as temporary:
            code="import os,signal;os.kill(os.getpid(),signal.SIGSEGV)"
            result=run_timed_command([sys.executable,"-c",code],2,Path(temporary)/"log")
        self.assertEqual(result["failure_category"],"crash");self.assertEqual(result["termination_signal"],signal.SIGSEGV)

if __name__=="__main__":unittest.main()
