#!/usr/bin/env python3
"""Reviewed outer launcher: authenticate worker and plan before any model process."""
import hashlib,subprocess,sys
from pathlib import Path
WORKER_SHA='2ac4e0c9610b2521784fea8b3ab7c664fcf91e6830e16434b4639de1a557b931'
PLAN_SHA='c7447c7150838dba7d98adf1a8dbca572b857511a341b505d4ac7e35077e1b02'

def verify_pair(worker,plan):
 for file,expected in [(worker,WORKER_SHA),(plan,PLAN_SHA)]:
  if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('Unreviewed execution identity: '+str(file))

def launch(root):
 out=root/'outputs/q2-block15-mlp81-v2';worker=out/'execution/worker.py';plan=out/'plan.json'
 verify_pair(worker,plan)
 return subprocess.run([str(root/'.venv/bin/python'),str(worker)],cwd=root,check=True).returncode
if __name__=='__main__':raise SystemExit(launch(Path.cwd()))
