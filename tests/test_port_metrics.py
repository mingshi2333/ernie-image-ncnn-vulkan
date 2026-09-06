import json, signal, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/"tools"))
import benchmark_pipeline
from benchmark_pipeline import run_timed_command
from port_metrics import build_protocol, paired_schedule, summarize_pairs, validate_measurement, _canonical, _digest

ROOT=Path(__file__).parents[1]

def synthetic_contract():
    protocol={"status":"frozen_inputs_no_results","performance_cases":6,
              "performance":{"measured_pairs":5,"warmups_per_port":1,"trace":False,
                             "order":["AB","BA","AB","BA","AB"]}}
    model={"official_revision":"fixture","weights":"same"}
    precision={"text":"fp32","dit":"bf16","scheduler":"fp32","vae":"fp32"}
    cases=[]
    for index in range(6):
        pe={"enabled":False}
        case={"id":f"performance-{index}","split":"performance",
              "prompt_sha256":f"{index + 1:064x}","noise_sha256":f"{index + 101:064x}",
              "noise_dtype":"<f4","shape":[512,512],"shape_order":"WH","steps":8,"cfg":1.0,
              "model_identity":model,"dtype_by_stage":precision,"pe":pe}
        if index==4:
            case["pe"]={"enabled":True,"sampling":"greedy","temperature":0,"top_p":1,
                        "max_input_tokens":2048,"max_new_tokens":2048,"stop_at_eos":True,
                        "add_generation_prompt":False,"template_source":"fixture",
                        "template_sha256":"a"*64}
        if index==5:
            case.update(mode="img2img",input_image_sha256="b"*64,decoded_rgb_sha256="c"*64,
                        strength=.5,resize_policy={"mode":"stretch"})
        cases.append(case)
    protocol_bytes=_canonical(protocol)+b"\n"
    manifest={"schema_version":1,"status":"frozen_inputs_no_formal_results","cases":cases,
              "files":[{"path":"protocol.json","sha256":_digest(protocol_bytes),"size_bytes":len(protocol_bytes)}]}
    manifest["manifest_sha256"]=_digest(_canonical(manifest))
    return manifest,protocol

MANIFEST,PROTOCOL=synthetic_contract()
CASES={c["id"]:c for c in MANIFEST["cases"] if c["split"]=="performance"}
FORMAL_CASE_IDS=tuple(CASES)
def side(case,seconds=2.0,device="vulkan"):
    start=1_000_000_000
    pe=json.loads(json.dumps(case["pe"]))
    return {"status":"ok","quality_status":"passed","scope":"end_to_end","trace":False,
            "input_id":case["prompt_sha256"],"noise_sha256":case["noise_sha256"],"noise_dtype":case["noise_dtype"],
            "model_id":_canonical(case["model_identity"]).decode(),"weights_canonical_sha256":"c"*64,
            "weight_identity_status":"proven","pe_identity":_canonical(pe).decode(),"pe":pe,
            "precision_by_stage":dict(case["dtype_by_stage"]),"device_by_stage":{k:device for k in case["dtype_by_stage"]},
            "shape":list(case["shape"]),"shape_order":case["shape_order"],"steps":case["steps"],"cfg":case["cfg"],
            "case_identity_sha256":_digest(_canonical(case)),
            "started_monotonic_ns":start,"finished_monotonic_ns":start+int(seconds*1e9),"wall_seconds":seconds,
            **({k:case[k] for k in ("input_image_sha256","decoded_rgb_sha256","strength","resize_policy")}
               if case.get("mode")=="img2img" else {})}
def pair(case="performance-0",index=0):
    frozen=CASES[case]
    return {"case_id":case,"pair_index":index,"phase":"measured","order":"AB" if index%2==0 else "BA",
            "trace":False,"candidate":side(frozen,2.0,"vulkan"),"reference":side(frozen,4.0,"cpu")}
def full_pairs(): return [pair(case,index) for case in FORMAL_CASE_IDS for index in range(5)]

class PortMetricsTest(unittest.TestCase):
    def test_default_unit_contract_is_version_control_only(self):
        source=Path(__file__).read_text()
        self.assertNotIn('outputs'+'/port-corpus-v1',source)
        self.assertEqual(_digest(_canonical({k:v for k,v in MANIFEST.items() if k!="manifest_sha256"})),
                         MANIFEST["manifest_sha256"])

    def test_frozen_denominator_completes_and_allows_different_devices(self):
        result=summarize_pairs(full_pairs(),MANIFEST,PROTOCOL)
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
                result=summarize_pairs(records,MANIFEST,PROTOCOL);self.assertEqual(result["status"],"incomplete")
                self.assertIsNone(result["geomean_ratio"])

    def test_failed_reference_is_not_infinite_speedup(self):
        records=full_pairs();records[0]["reference"]["status"]="oom";records[0]["reference"]["quality_status"]="unknown"
        result=summarize_pairs(records,MANIFEST,PROTOCOL)
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
        record=pair();validate_measurement(record,CASES[record["case_id"]])
        record=pair();record["reference"]["precision_by_stage"]["dit"]="fp16"
        with self.assertRaisesRegex(ValueError,"precision_by_stage mismatch"):validate_measurement(record)
        record=pair();record["reference"]["pe_identity"]="enabled:hash"
        with self.assertRaisesRegex(ValueError,"pe_identity mismatch"):validate_measurement(record)

    def test_complete_grid_with_wrong_frozen_shape_or_steps_is_incomplete(self):
        records=full_pairs()
        for record in records:
            record["candidate"]["shape"]=[64,64];record["candidate"]["steps"]=1
            record["reference"]["shape"]=[64,64];record["reference"]["steps"]=1
        result=summarize_pairs(records,MANIFEST,PROTOCOL)
        self.assertEqual(result["status"],"incomplete");self.assertIsNone(result["geomean_ratio"])

    def test_cases_cannot_reuse_one_input_or_disable_frozen_pe(self):
        records=full_pairs();first=CASES["performance-0"]
        for record in records:
            for name in ("candidate","reference"):
                record[name]["input_id"]=first["prompt_sha256"]
                record[name]["pe"]={"enabled":False};record[name]["pe_identity"]='{"enabled":false}'
        result=summarize_pairs(records,MANIFEST,PROTOCOL)
        self.assertEqual(result["status"],"incomplete");self.assertIsNone(result["geomean_ratio"])

    def test_img2img_rgb_strength_and_resize_are_bound(self):
        record=pair("performance-5");record["candidate"]["strength"]=0.1
        with self.assertRaisesRegex(ValueError,"strength"):validate_measurement(record,CASES["performance-5"])

    def test_missing_frozen_field_or_wrong_protocol_never_completes(self):
        records=full_pairs();del records[0]["candidate"]["cfg"]
        self.assertEqual(summarize_pairs(records,MANIFEST,PROTOCOL)["status"],"incomplete")
        protocol=json.loads(json.dumps(PROTOCOL));protocol["performance"]["measured_pairs"]=4
        with self.assertRaises(ValueError):summarize_pairs(full_pairs(),MANIFEST,protocol)

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

    def test_benchmark_rejects_unpaired_img2img_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"out"
            run=subprocess.run([sys.executable,str(Path(__file__).parents[1]/"tools"/"benchmark_pipeline.py"),
                "--model",str(Path(temporary)/"model"),"--output",str(output),"--input-image","x"],capture_output=True)
            self.assertEqual(run.returncode,2);self.assertIn(b'must be provided together',run.stderr)

    def test_benchmark_snapshots_and_binds_img2img_inputs(self):
        import hashlib
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);model=root/'model';model.mkdir();(model/'manifest.json').write_text('{}')
            runner=root/'runner';runner.write_bytes(b'runner');image=root/'input.png'
            Image.new('RGB',(2,1),(10,20,30)).save(image)
            raw=image.read_bytes();decoded=Image.open(image).convert('RGB').tobytes()
            sha=lambda value:hashlib.sha256(value).hexdigest()
            output=root/'output'
            argv=['benchmark_pipeline','--model',str(model),'--runner',str(runner),'--output',str(output),
                  '--device','cpu','--precision','fp32','--latent',str(root/'noise.f32'),
                  '--noise-sha256','noise','--input-image',str(image),'--strength','.5',
                  '--input-image-sha256',sha(raw),'--decoded-rgb-sha256',sha(decoded)]
            (root/'noise.f32').write_bytes(b'noise')
            def fake_timing(command,timeout,log):
                target=Path(command[command.index('--output')+1]);Image.new('RGB',(512,384)).save(target)
                return {'return_code':0,'failure_category':None,'wall_started_monotonic_ns':1,
                        'wall_finished_monotonic_ns':2,'wall_seconds':1e-9}
            with patch.object(sys,'argv',argv),\
                 patch.object(benchmark_pipeline,'verify_package',return_value=({'config':{'packed_width':32,'packed_height':24}},None)),\
                 patch('source_inventory.source_files',return_value=[]),\
                 patch.object(benchmark_pipeline,'run_timed_command',side_effect=fake_timing):
                self.assertEqual(benchmark_pipeline.main(),0)
            result=json.loads((output/'result.json').read_text())
            self.assertEqual(result['input_image_sha256'],sha(raw));self.assertEqual(result['decoded_rgb_sha256'],sha(decoded))
            self.assertEqual(result['strength'],.5);self.assertEqual(result['shape'],[512,384])
            self.assertIn('--input',result['command']);self.assertIn('--resize',result['command'])

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

    def test_positive_high_exit_status_does_not_invent_signal(self):
        with tempfile.TemporaryDirectory() as temporary:
            for code in (137,139,200):
                result=run_timed_command([sys.executable,"-c",f"raise SystemExit({code})"],2,Path(temporary)/f"log-{code}")
                self.assertEqual(result["failure_category"],"runtime_failure")
                self.assertIsNone(result["termination_signal"])

    def test_missing_sampler_still_writes_failure_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);model=root/"model";model.mkdir();(model/"manifest.json").write_text("{}")
            runner=root/"runner";runner.write_text("unused");output=root/"output"
            argv=["benchmark_pipeline","--model",str(model),"--runner",str(runner),"--output",str(output)]
            with patch.object(sys,"argv",argv),patch.object(benchmark_pipeline,"verify_package",return_value=({"config":{"packed_width":32,"packed_height":24}},None)),\
                 patch("source_inventory.source_files",return_value=[]),\
                 patch.object(benchmark_pipeline.subprocess,"Popen",side_effect=FileNotFoundError("synthetic nvidia-smi missing")):
                self.assertEqual(benchmark_pipeline.main(),1)
            result=json.loads((output/"result.json").read_text())
            self.assertEqual(result["status"],"incomplete");self.assertEqual(result["failure_category"],"runtime_failure")

if __name__=="__main__":unittest.main()
