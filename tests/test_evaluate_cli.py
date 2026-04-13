from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.eval_test_helpers import make_saved_run, write_history_jsonl


class TestEvaluateCLI(unittest.TestCase):
    def test_help_for_new_subcommands(self):
        for mode in ("metrics", "predictive-compare", "surface"):
            with self.subTest(mode=mode):
                proc = subprocess.run(
                    [sys.executable, "-m", "unified_stpp", "evaluate", mode, "--help"],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn(mode, proc.stdout)

    def test_metrics_cli_core_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = write_history_jsonl(root / "test.jsonl")
            run_dir = make_saved_run(root, preset="diffusion_stpp", label="diffusion")
            out_dir = root / "metrics_core"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unified_stpp",
                    "evaluate",
                    "metrics",
                    "--run",
                    str(run_dir),
                    "--data",
                    str(data_path),
                    "--metric-profile",
                    "core",
                    "--max-seqs",
                    "1",
                    "--max-events",
                    "3",
                    "--device",
                    "cpu",
                    "--out",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "evaluation_manifest.json").exists())
            self.assertTrue((out_dir / "artifacts.json").exists())

    def test_metrics_cli_explicit_heavy_metric_requires_matching_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = write_history_jsonl(root / "test.jsonl")
            run_dir = make_saved_run(root, preset="diffusion_stpp", label="diffusion")

            fail = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unified_stpp",
                    "evaluate",
                    "metrics",
                    "--run",
                    str(run_dir),
                    "--data",
                    str(data_path),
                    "--metric",
                    "temporal_crps",
                    "--max-seqs",
                    "1",
                    "--max-events",
                    "2",
                    "--k-pred",
                    "1",
                    "--device",
                    "cpu",
                    "--out",
                    str(root / "metrics_fail"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(fail.returncode, 0)
            self.assertIn("predictive_samples", fail.stderr)

            out_dir = root / "metrics_predictive_metric"
            ok = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unified_stpp",
                    "evaluate",
                    "metrics",
                    "--run",
                    str(run_dir),
                    "--data",
                    str(data_path),
                    "--metric-profile",
                    "predictive",
                    "--metric",
                    "temporal_crps",
                    "--max-seqs",
                    "1",
                    "--max-events",
                    "2",
                    "--k-pred",
                    "1",
                    "--device",
                    "cpu",
                    "--out",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertTrue((out_dir / "metrics.json").exists())

    def test_metrics_cli_predictive_artifact_reuse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = write_history_jsonl(root / "test.jsonl")
            run_dir = make_saved_run(root, preset="diffusion_stpp", label="diffusion")
            artifact_dir = root / "metric_artifacts"
            first_out = root / "metrics_predictive_first"
            second_out = root / "metrics_predictive_second"
            base_cmd = [
                sys.executable,
                "-m",
                "unified_stpp",
                "evaluate",
                "metrics",
                "--run",
                str(run_dir),
                "--data",
                str(data_path),
                "--metric-profile",
                "predictive",
                "--max-seqs",
                "1",
                "--max-events",
                "2",
                "--k-pred",
                "1",
                "--device",
                "cpu",
                "--artifact-dir",
                str(artifact_dir),
            ]

            first = subprocess.run(
                [*base_cmd, "--artifact-mode", "load_or_compute", "--out", str(first_out)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)

            second = subprocess.run(
                [*base_cmd, "--artifact-mode", "load_only", "--out", str(second_out)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)

            with open(first_out / "evaluation_manifest.json") as f:
                first_manifest = json.load(f)
            with open(second_out / "evaluation_manifest.json") as f:
                second_manifest = json.load(f)
            self.assertTrue(
                first_manifest["artifacts"]["events"]["predictive_samples"].startswith(
                    "computed_written:"
                )
            )
            self.assertTrue(
                second_manifest["artifacts"]["events"]["predictive_samples"].startswith(
                    "loaded:"
                )
            )

    def test_predictive_compare_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_path = write_history_jsonl(root / "history.jsonl")
            run_dir = make_saved_run(root, preset="diffusion_stpp", label="diffusion")
            out_dir = root / "predictive_cli"

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unified_stpp",
                    "evaluate",
                    "predictive-compare",
                    "--run",
                    str(run_dir),
                    "--label",
                    "Diffusion",
                    "--history",
                    str(history_path),
                    "--split",
                    "test",
                    "--seq-idx",
                    "0",
                    "--start-event-idx",
                    "1",
                    "--history-length",
                    "3",
                    "--horizon",
                    "0.4",
                    "--step-size",
                    "0.4",
                    "--n-frames",
                    "1",
                    "--n-rollouts",
                    "1",
                    "--grid-size",
                    "8",
                    "--bandwidth",
                    "0.25",
                    "--device",
                    "cpu",
                    "--seed",
                    "13",
                    "--out",
                    str(out_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "artifacts.json").exists())

    def test_surface_cli_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_path = write_history_jsonl(root / "test.jsonl")
            run_dir = make_saved_run(root, preset="auto_stpp", label="auto")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unified_stpp",
                    "evaluate",
                    "surface",
                    "--run",
                    str(run_dir),
                    "--history",
                    str(history_path),
                    "--seq-idx",
                    "0",
                    "--profile",
                    "notebook_faithful",
                    "--x-nstep",
                    "5",
                    "--y-nstep",
                    "5",
                    "--t-nstep",
                    "4",
                    "--frame-index",
                    "1",
                    "--device",
                    "cpu",
                    "--no-interactive",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            out_dir = run_dir / "evaluate" / "surface" / "notebook_faithful_test_seq000"
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "artifacts.json").exists())

    def test_split_must_match_inferable_history_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_path = write_history_jsonl(root / "train.jsonl")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unified_stpp",
                    "evaluate",
                    "surface",
                    "--run",
                    str(root / "missing_run"),
                    "--history",
                    str(history_path),
                    "--split",
                    "test",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("does not match", proc.stderr)


if __name__ == "__main__":
    unittest.main()
