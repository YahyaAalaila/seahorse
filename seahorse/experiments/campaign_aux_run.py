"""Serial CPU queue for frozen external baselines and native count controls.

Dry run is the default. Explicit execution enforces subprocess timeouts and
records every attempt; only validation-selected external trials open test data.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from seahorse.experiments.campaign import digest_file, verify, write_json


def select_trial(jobs, root):
    candidates=[]
    for job in jobs:
        out=root/"external_results"/job["id"]
        path=out/"status.json"
        if not path.exists(): raise ValueError("All external LR trials must be attempted before selection.")
        state=json.loads(path.read_text()).get("state")
        if state not in {"completed","failed","timeout"}: raise ValueError("An external trial is unfinished.")
        if state != "completed": continue
        metrics=json.loads((out/"metrics.json").read_text())
        value=-metrics["val_loglik"] if job["kind"] == "wsm" else metrics["best_validation_joint_nll"]
        if math.isfinite(value) and Path(metrics["checkpoint"]).is_file():
            candidates.append((float(value),job["lr_trial"],job["id"],metrics["checkpoint"]))
    if not candidates: raise ValueError("No finite validation checkpoint among the declared trials.")
    best=min(candidates)
    return next(j for j in jobs if j["id"] == best[2]),best[3],best[0]


def _attempt(command,out,limit,identity):
    path=out/"status.json"
    if path.exists(): return json.loads(path.read_text())
    out.mkdir(parents=True,exist_ok=True)
    status={"state":"running","started_utc":datetime.now(timezone.utc).isoformat(),**identity}
    write_json(path,status); started=time.monotonic()
    with (out/"stdout.log").open("w") as log:
        try:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=limit,check=False)
            status.update(state="completed" if result.returncode == 0 else "failed",returncode=result.returncode)
        except subprocess.TimeoutExpired:
            status.update(state="timeout",timeout_seconds=limit)
    status["elapsed_seconds"]=time.monotonic()-started
    write_json(path,status)
    return status


def queue(manifest_path,*,execute=False,phase="all",max_actions=None):
    manifest=verify(manifest_path); auxiliary=manifest["auxiliary"]
    root=Path(manifest["spec"]["auxiliary_manifest"]).parent
    jobs=auxiliary["external_jobs"]
    if not execute:
        return {"state":"dry_run","external_fit_specs":len(jobs),
                "validation_selected_evaluations":len(auxiliary["external_selected_evaluations"]),
                "count_jobs":len(auxiliary["native_count_jobs"]),"phase":phase}
    import fcntl
    executed=0; stop="queue_exhausted"
    def blocked(reason):
        write_json(root/"queue_state.json",{"state":reason,"executed":executed,"manifest_sha256":manifest["sha256"]})
        return {"state":reason,"executed":executed}
    def can_start(limit=660):
        if max_actions is not None and executed >= max_actions: return "requested_action_limit"
        if datetime.now(timezone.utc) >= datetime.fromisoformat(manifest["spec"]["compute"]["stop_before_utc"]): return "deadline_reached"
        used=sum(float(json.loads(p.read_text()).get("elapsed_seconds",0)) for p in root.rglob("status.json"))/3600
        if used+limit/3600 > manifest["spec"]["compute"]["auxiliary_cpu_process_hours"]: return "cpu_process_budget_exhausted"
        return None
    with (root/"queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if phase in {"all","external"}:
            for job in sorted(jobs,key=lambda j:(j["lr_trial"],j["id"])):
                out=root/"external_results"/job["id"]
                if (out/"status.json").exists(): continue
                if (stop:=can_start()): return blocked(stop)
                if digest_file(job["spec"]) != job["spec_sha256"]: raise ValueError("External spec changed after freeze.")
                spec=json.loads(Path(job["spec"]).read_text())
                command=[sys.executable,*job["fit_command"][1:]]
                _attempt(command,out,spec["resource_policy"]["outer_watchdog_seconds"],{"job_id":job["id"],"manifest_sha256":manifest["sha256"]})
                executed+=1
            for group in sorted({j["selection_group"] for j in jobs}):
                out=root/"external_results"/group/"evaluate"
                if (out/"status.json").exists(): continue
                if (stop:=can_start(1800)): return blocked(stop)
                try:
                    chosen,checkpoint,value=select_trial([j for j in jobs if j["selection_group"]==group],root)
                except ValueError as exc:
                    write_json(out/"status.json",{"state":"failed","reason":str(exc),"selection_group":group});continue
                write_json(out/"validation_selection.json",{"selected_job":chosen["id"],"validation_objective":value,"checkpoint":checkpoint})
                command=[sys.executable,"-m","seahorse.baselines.run","--spec",chosen["spec"],"--checkpoint",checkpoint,"--out",str(out),"--stage","evaluate"]
                _attempt(command,out,1800,{"selection_group":group,"selected_job":chosen["id"],"manifest_sha256":manifest["sha256"]});executed+=1
        if phase in {"all","counts"}:
            for job in auxiliary["native_count_jobs"]:
                out=root/"count_results"/job["id"]
                if (out/"status.json").exists(): continue
                if (stop:=can_start(7200)): return blocked(stop)
                for split,path in job["route"]["splits"].items():
                    if digest_file(path) != job["route"]["split_sha256"][split]: raise ValueError("Count input changed after freeze.")
                _attempt([sys.executable,*job["command"][1:]],out,7200,{"job_id":job["id"],"manifest_sha256":manifest["sha256"]});executed+=1
    return blocked("queue_exhausted")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--execute",action="store_true")
    p.add_argument("--phase",choices=("external","counts","all"),default="all");p.add_argument("--max-actions",type=int)
    args=p.parse_args()
    print(json.dumps(queue(args.manifest,execute=args.execute,phase=args.phase,max_actions=args.max_actions),indent=2))


if __name__ == "__main__": main()
