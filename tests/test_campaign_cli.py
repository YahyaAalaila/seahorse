import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import numpy as np
import scipy

from seahorse import __main__


def test_campaign_is_listed_in_public_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["python -m seahorse", "--help"])
    with pytest.raises(SystemExit) as exited:
        __main__.main()
    assert exited.value.code == 0
    assert "campaign" in capsys.readouterr().out


def test_campaign_queue_dry_run_delegates_without_execution(monkeypatch, capsys):
    called = {}

    def fake_queue(manifest, *, execute, stage, max_jobs, phase, runner, models):
        called.update(
            manifest=manifest,
            execute=execute,
            stage=stage,
            max_jobs=max_jobs,
            phase=phase,
            runner=runner,
            models=models,
        )
        return {"state": "dry_run", "pending_actions": 3}

    from seahorse.experiments import campaign_run

    monkeypatch.setattr(campaign_run, "queue", fake_queue)
    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m seahorse", "campaign", "queue", "--manifest", "campaign/manifest.json"],
    )
    __main__.main()
    assert called == {
        "manifest": "campaign/manifest.json",
        "execute": False,
        "stage": "core",
        "max_jobs": None,
        "phase": "all",
        "runner": "all",
        "models": None,
    }
    assert "dry_run" in capsys.readouterr().out


def test_campaign_preflight_exposes_runner_subset(monkeypatch, capsys):
    called = {}

    def fake_preflight(manifest, **kwargs):
        called.update(manifest=manifest, **kwargs)
        return {"ready": False}

    from seahorse.experiments import campaign_preflight

    monkeypatch.setattr(campaign_preflight, "run_preflight", fake_preflight)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m seahorse",
            "campaign",
            "preflight",
            "--manifest",
            "m.json",
            "--runner",
            "statstpp",
            "--device",
            "cpu",
            "--skip-dataset-smokes",
        ],
    )
    __main__.main()
    assert called == {
        "manifest": "m.json",
        "device": "cpu",
        "models": None,
        "dataset_smokes": False,
        "runner": "statstpp",
    }
    assert "false" in capsys.readouterr().out


def test_preflight_partial_runner_smokes_merge_and_runtime_only_upgrades(monkeypatch):
    from seahorse.experiments import campaign_preflight as preflight

    manifest = {
        "spec": {
            "models": [{"id": "neural", "runner": "nnstpp"}, {"id": "stat", "runner": "statstpp"}]
        }
    }
    evidence = {"models": {"neural": {}}}
    first = preflight._merge_dataset_smokes({}, {"nnstpp": {"status": "pass"}}, manifest, evidence)
    assert first["status"] == "pass"
    evidence["models"]["stat"] = {}
    merged = preflight._merge_dataset_smokes(
        first, {"statstpp": {"status": "pass"}}, manifest, evidence
    )
    assert merged["status"] == "pass"
    assert set(merged["runner_smokes"]) == {"nnstpp", "statstpp"}

    runtime = {
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": "old",
        "device": "cpu",
        "gpu": None,
    }
    monkeypatch.setattr(preflight.torch.cuda, "get_device_name", lambda _: "fixture-gpu")
    preflight._upgrade_runtime(runtime, device="cuda")
    assert runtime["device"] == "cuda" and runtime["gpu"] == "fixture-gpu"
    preflight._upgrade_runtime(runtime, device="cpu")
    assert runtime["device"] == "cuda"  # later stat-only evidence cannot downgrade it


def test_preflight_full_pair_smokes_record_per_model_gates():
    from seahorse.experiments import campaign_preflight as preflight

    manifest = {
        "spec": {"models": [{"id": "a", "runner": "nnstpp"}, {"id": "b", "runner": "statstpp"}]}
    }
    evidence = {"models": {"a": {}, "b": {}}}
    entry = preflight._full_dataset_smoke(
        {},
        {
            "nnstpp": {"a": {"status": "pass", "trainer_check": "a"}},
            "statstpp": {"b": {"status": "fail", "trainer_check": "b"}},
        },
        manifest,
        evidence,
    )
    assert entry["status"] == "fail"
    assert entry["runner_smokes"]["nnstpp"]["model_checks"]["a"]["status"] == "pass"
    assert entry["runner_smokes"]["statstpp"]["model_checks"]["b"]["status"] == "fail"
    from seahorse.experiments.campaign_run import _dataset_gate

    assert _dataset_gate({"datasets": {"data": entry}}, "data", "nnstpp", "a")
    assert not _dataset_gate({"datasets": {"data": entry}}, "data", "statstpp", "b")
    manifest["spec"]["models"].append({"id": "c", "runner": "nnstpp"})
    partial = preflight._full_dataset_smoke(
        {}, {"nnstpp": {"a": {"status": "pass"}}}, manifest, evidence
    )
    assert partial["runner_smokes"]["nnstpp"]["status"] == "fail"


def test_campaign_detach_starts_new_session_and_records_receipt(monkeypatch, tmp_path):
    from seahorse.cli import campaign

    captured = {}

    class Process:
        pid = 4242

    def fake_popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return Process()

    monkeypatch.setattr(campaign.subprocess, "Popen", fake_popen)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = SimpleNamespace(
        manifest=str(manifest), stage="core", phase="fit", runner="statstpp", max_jobs=2, watch=True
    )
    result = campaign._start_detached_queue(args, manifest={}, summary={"state": "dry_run"})
    assert result["state"] == "detached" and result["pid"] == 4242
    assert "--detach" not in captured["command"]
    assert "--watch" not in captured["command"]
    assert "--watch-worker" in captured["command"]
    assert captured["start_new_session"] is True
    assert captured["stdin"] is campaign.subprocess.DEVNULL
    receipt = tmp_path / "queue_statstpp.pid.json"
    assert receipt.exists() and "4242" in receipt.read_text()


def test_campaign_subset_detach_forwards_models_and_has_own_receipts(monkeypatch, tmp_path):
    from seahorse.cli import campaign

    captured = {}

    class Process:
        pid = 4343

    monkeypatch.setattr(
        campaign.subprocess,
        "Popen",
        lambda command, **kwargs: captured.update(command=command, **kwargs) or Process(),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = SimpleNamespace(
        manifest=str(manifest),
        stage="core",
        phase="fit",
        runner="statstpp",
        max_jobs=None,
        watch=False,
        models=["hawkes_exp_gaussian"],
    )
    result = campaign._start_detached_queue(args, manifest={}, summary={})
    assert ["--models", "hawkes_exp_gaussian"] == captured["command"][
        captured["command"].index("--models") :
    ]
    assert "models_" in Path(result["receipt"]).name
    assert "models_" in Path(result["log"]).name


def test_campaign_run_dispatches_in_process(monkeypatch):
    from seahorse.experiments import campaign_run

    called = {}
    monkeypatch.setattr(campaign_run, "main", lambda argv: called.setdefault("argv", argv))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m seahorse",
            "campaign",
            "run",
            "--manifest",
            "m.json",
            "--job",
            "cell",
            "--device",
            "cpu",
        ],
    )
    __main__.main()
    assert called["argv"] == ["run", "--manifest", "m.json", "--job", "cell", "--device", "cpu"]
