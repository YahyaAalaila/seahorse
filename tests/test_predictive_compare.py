from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from unified_stpp.evaluation import (
    ExactProposalConfig,
    FrameWindow,
    HistoryQuery,
    PredictiveComparator,
    PredictiveCompareSpec,
    PredictiveComparisonResult,
    PredictiveFrameResult,
    PredictiveModelResult,
    RunTarget,
    load_predictive_bundle,
    write_predictive_bundle,
)
from unified_stpp.evaluation.common import derive_seed, load_runs, load_sequence, resolve_device, slice_initial_history
from unified_stpp.evaluation.predictive_sampling import evaluate_teacher_forced_frame
from unified_stpp.viz import PredictiveRenderConfig, render_predictive_bundle

from tests.eval_test_helpers import assert_finite_array, make_saved_run, write_history_jsonl


class TestPredictiveSeedHelpers(unittest.TestCase):
    def test_derive_seed_is_stable_and_sensitive(self):
        seed_a = derive_seed(7, "model-a", 0, 1)
        seed_b = derive_seed(7, "model-a", 0, 1)
        seed_c = derive_seed(7, "model-a", 0, 2)
        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, seed_c)


class TestPredictiveBundleIO(unittest.TestCase):
    def test_roundtrip_preserves_primary_sample_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = PredictiveComparisonResult(
                history_path=Path("/tmp/history.jsonl"),
                split="test",
                seq_idx=0,
                start_event_idx=1,
                initial_history_length=2,
                sequence_length=5,
                sequence_start_time=0.1,
                sequence_end_time=1.2,
                spec=PredictiveCompareSpec(
                    n_frames=1,
                    horizon=0.5,
                    step_size=0.5,
                    n_rollouts=2,
                    grid_size=4,
                    bandwidth=0.2,
                    exact_proposal=ExactProposalConfig(),
                    seed=11,
                ),
                xs=np.linspace(0.0, 1.0, 4, dtype=np.float32),
                ys=np.linspace(0.0, 1.0, 4, dtype=np.float32),
                color_scale={"vmin": 0.0, "vmax": 1.0},
                frame_schedule=[FrameWindow(index=0, start=0.3, end=0.8)],
                models=[
                    PredictiveModelResult(
                        label="toy",
                        safe_label="toy",
                        preset="auto_stpp",
                        preset_status="canonical",
                        nll_kind="exact",
                        nll_report_space="raw",
                        run_dir=Path("/tmp/run"),
                        sampling_backend="external_thinning_rollout",
                        frames=[
                            PredictiveFrameResult(
                                window=FrameWindow(index=0, start=0.3, end=0.8),
                                history_locs=np.asarray([[0.1, 0.2]], dtype=np.float32),
                                pooled_event_times=np.asarray([0.4, 0.6], dtype=np.float32),
                                pooled_event_locs=np.asarray([[0.2, 0.3], [0.4, 0.7]], dtype=np.float32),
                                rollout_event_counts=np.asarray([1, 1], dtype=np.int32),
                                true_event_times=np.asarray([0.5], dtype=np.float32),
                                true_event_locs=np.asarray([[0.3, 0.4]], dtype=np.float32),
                                mean_events_per_rollout=1.0,
                                derived_kde_rate_surface=np.ones((4, 4), dtype=np.float32),
                                diagnostics={"bridge_rejects": 0},
                            )
                        ],
                    )
                ],
                seed_policy={"base_seed": 11, "derivation": "stable", "caveat": "test"},
            )
            out_dir = Path(tmpdir) / "bundle"
            write_predictive_bundle(out_dir, result)
            loaded = load_predictive_bundle(out_dir)

            self.assertEqual(loaded.seed_policy["base_seed"], 11)
            self.assertEqual(len(loaded.models), 1)
            np.testing.assert_allclose(
                loaded.models[0].frames[0].pooled_event_times,
                result.models[0].frames[0].pooled_event_times,
            )
            np.testing.assert_allclose(
                loaded.models[0].frames[0].pooled_event_locs,
                result.models[0].frames[0].pooled_event_locs,
            )


class TestPredictiveComparatorSmoke(unittest.TestCase):
    def test_compare_bundle_and_render_with_native_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_path = write_history_jsonl(root / "history.jsonl")
            run_dir = make_saved_run(root, preset="diffusion_stpp", label="diffusion")

            result = PredictiveComparator().compare(
                [
                    RunTarget(run=run_dir, label="Diffusion"),
                ],
                HistoryQuery(
                    history_path=history_path,
                    split="test",
                    seq_idx=0,
                    start_event_idx=1,
                    history_length=3,
                ),
                PredictiveCompareSpec(
                    rollout_mode="teacher_forced",
                    n_frames=1,
                    horizon=0.4,
                    step_size=0.4,
                    n_rollouts=1,
                    grid_size=8,
                    bandwidth=0.25,
                    max_events_per_window=3,
                    bridge_retries=4,
                    exact_proposal=ExactProposalConfig(),
                    seed=17,
                    device="cpu",
                ),
            )

            self.assertEqual(len(result.models), 1)
            self.assertEqual(result.seed_policy["base_seed"], 17)
            self.assertEqual(result.models[0].label, "Diffusion")
            for model in result.models:
                self.assertEqual(len(model.frames), 1)
                frame = model.frames[0]
                assert_finite_array(self, frame.rollout_event_counts)
                assert_finite_array(self, frame.derived_kde_rate_surface)
                self.assertEqual(frame.derived_kde_rate_surface.shape, (8, 8))
                self.assertEqual(frame.rollout_event_counts.shape, (1,))

            out_dir = root / "predictive_out"
            write_predictive_bundle(out_dir, result)
            loaded = load_predictive_bundle(out_dir)
            artifacts = render_predictive_bundle(
                loaded,
                out_dir,
                PredictiveRenderConfig(plot_style="2d", fps=2.0, write_gif=False),
            )

            self.assertTrue((out_dir / "summary.json").exists())
            self.assertTrue((out_dir / "derived_surfaces.npz").exists())
            self.assertIn("predictive_panel_frame_000", artifacts)
            self.assertTrue(artifacts["predictive_panel_frame_000"].exists())

    def test_native_diffusion_backend_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_path = write_history_jsonl(root / "history.jsonl")
            run_dir = make_saved_run(root, preset="diffusion_stpp", label="diffusion")
            device = resolve_device("cpu")
            loaded = load_runs([run_dir], device=device, supported_presets={"diffusion_stpp"})[0]
            seq = load_sequence(history_path, 0)
            initial_history = slice_initial_history(seq, 1)
            frame = evaluate_teacher_forced_frame(
                loaded,
                seq=seq,
                window=FrameWindow(
                    index=0,
                    start=float(initial_history["times"][-1]),
                    end=float(initial_history["times"][-1] + 0.4),
                ),
                history_length=3,
                n_rollouts=1,
                xmin=0.0,
                xmax=1.0,
                ymin=0.0,
                ymax=1.0,
                lambda_bar=5.0,
                max_events_per_window=3,
                bridge_retries=4,
                adaptive_thinning=True,
                exact_proposal=ExactProposalConfig(),
                device=device,
                base_seed=23,
            )
            assert_finite_array(self, frame["rollout_event_counts"])
            assert_finite_array(self, frame["pooled_event_times"])
            assert_finite_array(self, frame["pooled_event_locs"])


if __name__ == "__main__":
    unittest.main()
