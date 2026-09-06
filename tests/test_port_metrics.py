import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from port_metrics import build_protocol, paired_schedule, summarize_pairs, validate_measurement


def pair(case_id="case", **updates):
    value={"case_id":case_id,"input_id":"input","noise_id":"noise","model_id":"model",
           "precision":"fp16","pe_enabled":False,"trace":False,"scope":"end_to_end",
           "candidate_seconds":2.0,"reference_seconds":4.0,
           "candidate_status":"ok","reference_status":"ok",
           "candidate_quality_status":"passed","reference_quality_status":"passed",
           "candidate_started_monotonic_ns":1_000_000_000,"candidate_finished_monotonic_ns":3_000_000_000,
           "reference_started_monotonic_ns":4_000_000_000,"reference_finished_monotonic_ns":8_000_000_000}
    value.update(updates)
    return value


class PortMetricsTest(unittest.TestCase):
    def test_complete_pairs_report_speed_ratio(self):
        result=summarize_pairs([pair("a"),pair("b",candidate_seconds=4,reference_seconds=8,
            candidate_finished_monotonic_ns=5_000_000_000,reference_finished_monotonic_ns=12_000_000_000)])
        self.assertEqual(result["status"],"complete")
        self.assertEqual(result["geomean_ratio"],2.0)
        self.assertEqual(result["candidate_median_seconds"],3.0)

    def test_failed_reference_is_not_infinite_speedup(self):
        value=pair("oom",reference_seconds=None,reference_status="oom")
        result=summarize_pairs([value])
        self.assertIsNone(result["geomean_ratio"])
        self.assertEqual(result["unavailable_case_ids"],["oom"])

    def test_trace_or_component_only_timing_is_rejected(self):
        for value in (pair(trace=True),pair(scope="dit_only")):
            with self.assertRaises(ValueError):validate_measurement(value)

    def test_missing_pe_or_identity_mismatch_is_rejected(self):
        missing=pair();del missing["pe_enabled"]
        with self.assertRaises(ValueError):validate_measurement(missing)
        mismatch=pair(candidate={"status":"ok","wall_seconds":2,"scope":"end_to_end","noise_id":"other"},
                      reference={"status":"ok","wall_seconds":3,"scope":"end_to_end"})
        with self.assertRaises(ValueError):validate_measurement(mismatch)

    def test_missing_quality_or_complete_clock_is_rejected(self):
        for value in (pair(candidate_quality_status=None),pair(reference_finished_monotonic_ns=None)):
            with self.assertRaises(ValueError):validate_measurement(value)

    def test_one_missing_case_invalidates_full_denominator(self):
        result=summarize_pairs([pair("good"),pair("missing",candidate_seconds=None)])
        self.assertEqual(result["status"],"incomplete")
        self.assertIsNone(result["geomean_ratio"])
        self.assertEqual(result["valid_pair_count"],1)

    def test_schedule_has_warmups_and_five_alternating_pairs(self):
        schedule=paired_schedule("x")
        self.assertEqual(len(schedule),12)
        self.assertEqual([r["port"] for r in schedule[:2]],["candidate","reference"])
        measured=schedule[2:]
        self.assertEqual([r["order"] for r in measured[::2]],["AB","BA","AB","BA","AB"])
        self.assertTrue(all(r["new_process"] for r in schedule))

    def test_budget_without_empirical_calibration_stays_incomplete(self):
        protocol=build_protocol([{"id":"512"}],None,10**9)
        self.assertEqual(protocol["status"],"incomplete")
        self.assertIsNone(protocol["budget"]["estimated_hours"])

    def test_benchmark_reports_unsupported_img2img_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"out"
            run=subprocess.run([sys.executable,str(Path(__file__).parents[1]/"tools"/"benchmark_pipeline.py"),
                "--model",str(Path(temporary)/"model"),"--output",str(output),
                "--input-image",str(Path(temporary)/"input.png")],capture_output=True,text=True)
            self.assertEqual(run.returncode,2)
            result=json.loads((output/"result.json").read_text())
            self.assertEqual(result["status"],"incomplete")
            self.assertIn("--input-image",result["unsupported_arguments"])


if __name__ == "__main__":unittest.main()
