from __future__ import annotations
import json
import seahorse.experiments.campaign_slurm as slurm


def _manifest():
    return {
        "sha256": "frozen",
        "spec": {"compute": {}},
        "jobs": [
            {
                "id": "d__njsde__hpo0",
                "model": "njsde",
                "action": "hpo",
                "runner": "nnstpp",
                "dependencies": [],
            },
            {
                "id": "d__njsde__seed1",
                "model": "njsde",
                "action": "refit",
                "runner": "nnstpp",
                "dependencies": ["d__njsde__hpo0"],
            },
            {
                "id": "d__p__seed1",
                "model": "homogeneous_poisson",
                "action": "refit",
                "runner": "statstpp",
                "dependencies": [],
            },
        ],
    }


def _plan(tmp_path, monkeypatch, **kwargs):
    p = tmp_path / "manifest.json"
    p.write_text("{}")
    monkeypatch.setattr(slurm, "verify", lambda _: _manifest())
    return slurm.build_plan(
        p, out=tmp_path / "plan.json", python="/env/python", **kwargs
    ), tmp_path / "plan.json"


def test_plan_uses_real_partitions_quotes_and_eval_command(tmp_path, monkeypatch):
    plan, _ = _plan(tmp_path, monkeypatch, container_prefix=["apptainer", "exec", "image file.sif"])
    h = next(x for x in plan["tasks"] if x["job_id"].endswith("hpo0"))
    e = next(x for x in plan["tasks"] if x["action"] == "evaluate")
    assert (
        "A100-80GB" in h["resource"]
        if "resource" in h
        else "A100-80GB" in plan["resource_config"]["heavy"]["partitions"]
    )
    assert "evaluate" in e["command"]
    cmd = slurm._sbatch(plan, h)
    assert cmd[-2] == "--wrap" and "'image file.sif'" in cmd[-1]


def test_dry_dispatch_and_eval_only_existing_completed_refit(tmp_path, monkeypatch):
    plan, path = _plan(tmp_path, monkeypatch, phase="evaluate")
    out = tmp_path / "jobs" / "d__njsde__seed1"
    out.mkdir(parents=True)
    (out / "status.json").write_text(json.dumps({"state": "completed"}))
    result = slurm.dispatch(path)
    assert result["state"] == "dry_run" and any(
        x["task_id"].endswith("evaluate") for x in result["launched"]
    )


def test_reconcile_hardkill_finalizes_missing_worker(tmp_path, monkeypatch):
    plan, path = _plan(tmp_path, monkeypatch)
    task = plan["tasks"][0]
    r = slurm._receipt(plan, task)
    r.parent.mkdir(parents=True)
    r.write_text(json.dumps({"slurm_job_id": "10", "scheduler_state": "RUNNING"}))

    class R:
        stdout = "10|OUT_OF_MEMORY|oom-kill|0:9|00:01:00|n1|gres/gpu=1\n"

    slurm.reconcile(path, execute=True, runner=lambda *a, **k: R())
    assert (
        json.loads((tmp_path / "jobs" / task["job_id"] / "status.json").read_text())[
            "failure_class"
        ]
        == "scheduler_out_of_memory"
    )


def test_global_concurrency_dry_plan(tmp_path, monkeypatch):
    plan, path = _plan(tmp_path, monkeypatch)
    plan["tasks"] += [
        {**plan["tasks"][0], "task_id": "extra0"},
        {**plan["tasks"][0], "task_id": "extra1"},
        {**plan["tasks"][0], "task_id": "extra2"},
    ]
    plan["sha256"] = slurm.canonical_hash({k: v for k, v in plan.items() if k != "sha256"})
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    assert (
        len(
            [
                x
                for x in slurm.dispatch(path)["launched"]
                if "njsde" in x["task_id"] or x["task_id"].startswith("extra")
            ]
        )
        == 2
    )


def test_submission_failure_keeps_durable_intent_without_resubmit(tmp_path, monkeypatch):
    plan, path = _plan(tmp_path, monkeypatch)

    class Fail:
        stdout = ""

    def runner(cmd, **kwargs):
        if cmd[0] == "sacct":
            return Fail()
        raise subprocess.CalledProcessError(1, cmd)

    import subprocess

    try:
        slurm.dispatch(path, execute=True, runner=runner)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("sbatch failure must propagate")
    receipt = slurm._receipt(plan, plan["tasks"][0])
    assert json.loads(receipt.read_text())["scheduler_state"] == "SUBMISSION_INTENT"


def test_dry_dispatch_is_read_only(tmp_path, monkeypatch):
    plan, path = _plan(tmp_path, monkeypatch)
    before = sorted(str(x.relative_to(tmp_path)) for x in tmp_path.rglob("*"))
    slurm.dispatch(path, execute=False)
    after = sorted(str(x.relative_to(tmp_path)) for x in tmp_path.rglob("*"))
    assert after == before


def test_multitick_hpo_refit_eval_dag_once_without_premature_eval_block(tmp_path, monkeypatch):
    plan, path = _plan(tmp_path, monkeypatch, runner="nnstpp")
    calls = []
    next_id = [100]

    class R:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "sacct":
            rows = []
            for jid in cmd[4].split(","):
                rows.append(f"{jid}|COMPLETED|None|0:0|00:01|n1|gres/gpu=1")
            return R("\n".join(rows) + "\n")
        if cmd[0] == "squeue":
            return R("")
        if cmd[0] == "sbatch":
            answer = str(next_id[0])
            next_id[0] += 1
            return R(answer + ";cluster\n")
        raise AssertionError(cmd)

    # Tick 1: HPO only; evaluation must merely wait.
    slurm.dispatch(path, execute=True, runner=fake)
    assert len([c for c in calls if c[0] == "sbatch"]) == 1
    assert not (tmp_path / "jobs" / "d__njsde__seed1" / "evaluation_status.json").exists()
    # HPO worker reaches its own terminal receipt, permitting the refit on tick 2.
    hpo = tmp_path / "jobs" / "d__njsde__hpo0"
    hpo.mkdir(parents=True)
    (hpo / "status.json").write_text(json.dumps({"state": "completed"}))
    slurm.dispatch(path, execute=True, runner=fake)
    assert len([c for c in calls if c[0] == "sbatch"]) == 2
    # Refit terminal permits evaluation on tick 3.
    refit = tmp_path / "jobs" / "d__njsde__seed1"
    refit.mkdir(parents=True)
    (refit / "status.json").write_text(json.dumps({"state": "completed"}))
    slurm.dispatch(path, execute=True, runner=fake)
    assert len([c for c in calls if c[0] == "sbatch"]) == 3
    assert not (refit / "evaluation_status.json").exists()
