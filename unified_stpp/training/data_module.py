"""
PyTorch Lightning DataModule for STPP datasets.
"""
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

from unified_stpp.data import STPPDataset, collate_fn
from unified_stpp.data.dataset import PaperSlidingWindowDataset


# ---------------------------------------------------------------------------
# Protocol compatibility guard
# ---------------------------------------------------------------------------

_SLIDING_WINDOW_VALID_PRESETS = frozenset({"auto_stpp", "deep_stpp"})


def assert_protocol_model_compatible(protocol: str, model_preset: str) -> None:
    """
    Raise if the requested protocol has not been validated for model_preset.

    ``"sliding_window"`` matches the AutoSTPP repo pipeline (MinMax +
    sliding windows) and has only been verified for auto_stpp / deep_stpp on
    synthetic STHP data.  Any other model would silently receive a very
    different coordinate system without this guardrail.
    """
    if (
        protocol == "sliding_window"
        and model_preset not in _SLIDING_WINDOW_VALID_PRESETS
    ):
        raise ValueError(
            f"protocol={protocol!r} has not been validated for model preset "
            f"{model_preset!r}.  Validated presets: "
            f"{sorted(_SLIDING_WINDOW_VALID_PRESETS)}.  "
            "Switch to protocol='standard' or add the preset to "
            "_SLIDING_WINDOW_VALID_PRESETS after explicit parity verification."
        )


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class STPPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_seqs,
        val_seqs,
        test_seqs=None,
        batch_size: int = 32,
        num_workers: int = 0,
        normalize: bool = True,
        seed: int = 42,
        # Data protocol options
        protocol: str = "standard",
        raw_seq: Optional[Dict] = None,
        paper_lookback: int = 10,
        paper_lookahead: int = 1,
        paper_split_ratio: Tuple[int, int, int] = (8, 1, 1),
        # Protocol guard: when set, setup() validates that protocol is compatible
        # with this preset before building datasets.
        model_preset: str = "",
    ):
        super().__init__()
        self.train_seqs = train_seqs
        self.val_seqs = val_seqs
        self.test_seqs = test_seqs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.protocol = protocol
        self.raw_seq = raw_seq
        self.paper_lookback = paper_lookback
        self.paper_lookahead = paper_lookahead
        self.paper_split_ratio = paper_split_ratio
        self.model_preset = model_preset
        # Populated by setup()
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        # Persistent generator for the train DataLoader.  Using a *single
        # Generator object* (not recreating it each call) means the shuffle
        # sequence advances deterministically across epochs while remaining
        # fully independent of the global torch RNG — eliminating the primary
        # source of run-to-run non-reproducibility on CPU.
        self._train_generator = None

    def setup(self, stage=None):
        # Guard so Lightning's repeated setup() calls don't recreate datasets.
        if self._train_dataset is not None:
            return

        if self.model_preset:
            assert_protocol_model_compatible(self.protocol, self.model_preset)

        if self.protocol == "sliding_window":
            self._setup_sliding_window()
        else:
            self._setup_standard()

    def _setup_standard(self):
        self._train_dataset = STPPDataset(
            self.train_seqs,
            normalize_time=self.normalize,
            normalize_space=self.normalize,
        )
        # Use training stats for val/test normalization
        self._val_dataset = STPPDataset(
            self.val_seqs,
            normalize_time=self.normalize,
            normalize_space=self.normalize,
            cov_mean=self._train_dataset.cov_mean,
            cov_std=self._train_dataset.cov_std,
        )
        # Override time/location normalization stats with training stats
        self._val_dataset.time_mean = self._train_dataset.time_mean
        self._val_dataset.time_std = self._train_dataset.time_std
        self._val_dataset.loc_mean = self._train_dataset.loc_mean
        self._val_dataset.loc_std = self._train_dataset.loc_std

        if self.test_seqs is not None:
            self._test_dataset = STPPDataset(
                self.test_seqs,
                normalize_time=self.normalize,
                normalize_space=self.normalize,
                cov_mean=self._train_dataset.cov_mean,
                cov_std=self._train_dataset.cov_std,
            )
            self._test_dataset.time_mean = self._train_dataset.time_mean
            self._test_dataset.time_std = self._train_dataset.time_std
            self._test_dataset.loc_mean = self._train_dataset.loc_mean
            self._test_dataset.loc_std = self._train_dataset.loc_std

        # Create the persistent generator once so its state evolves across
        # epochs, giving a different (but deterministic) shuffle each epoch.
        self._train_generator = torch.Generator()
        self._train_generator.manual_seed(self.seed)

    def _setup_sliding_window(self):
        """
        Build datasets matching the AutoSTPP paper pipeline for STHP:

        1. Convert absolute times → delta_t (delta_t[0] = 0).
        2. Stack [x, y, delta_t] and fit ONE global MinMaxScaler on ALL events
           (full sequence, intentional data leakage matching the original repo).
        3. Create sliding windows of shape (W, lookback+lookahead, 3).
        4. Split windows by index ratio [8, 1, 1].
        5. Wrap each split in PaperSlidingWindowDataset (same collate_fn API).
        """
        from sklearn.preprocessing import MinMaxScaler

        if self.raw_seq is None:
            raise ValueError(
                "STPPDataModule: protocol='sliding_window' requires raw_seq."
            )

        seq = self.raw_seq
        times_abs = np.asarray(seq["times"], dtype=np.float64)
        locs      = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, 2)

        # Convert absolute times → delta_t; set delta_t[0] = 0
        delta_t    = np.diff(times_abs, prepend=times_abs[0])
        delta_t[0] = 0.0

        # Stack [x, y, delta_t] — shape (N, 3)
        xyt = np.column_stack([locs[:, 0], locs[:, 1], delta_t]).astype(np.float32)

        # Fit MinMaxScaler on ALL events (matches original repo: global scaler)
        scaler   = MinMaxScaler()
        xyt_mm   = scaler.fit_transform(xyt).astype(np.float32)  # (N, 3)

        # Create sliding windows
        lb = self.paper_lookback
        la = self.paper_lookahead
        T  = lb + la
        N  = len(xyt_mm)
        W  = max(0, N - T + 1)
        windows = np.stack([xyt_mm[i : i + T] for i in range(W)], axis=0)  # (W, T, 3)

        # Split by ratio
        r0, r1, r2 = self.paper_split_ratio
        total_r    = r0 + r1 + r2
        n_tr = int(W * r0 / total_r)
        n_va = int(W * r1 / total_r)
        n_te = W - n_tr - n_va

        w_train = windows[:n_tr]
        w_val   = windows[n_tr : n_tr + n_va]
        w_test  = windows[n_tr + n_va :]

        self._train_dataset = PaperSlidingWindowDataset(w_train, scaler=scaler)
        self._val_dataset   = PaperSlidingWindowDataset(w_val,   scaler=scaler)
        self._test_dataset  = PaperSlidingWindowDataset(w_test,  scaler=scaler)

        # One-line audit print
        mn = np.round(scaler.data_min_, 4).tolist()
        mx = np.round(scaler.data_max_, 4).tolist()
        print(
            f"[DataModule] protocol={self.protocol!r}  scaler=MinMaxScaler"
            f"(min={mn}  max={mx})  time_format=delta_t→cumsum_mm  "
            f"window_shape=(B, {T}, 3)  "
            f"n_train={n_tr}  n_val={n_va}  n_test={n_te}"
        )

        self._train_generator = torch.Generator()
        self._train_generator.manual_seed(self.seed)

    @classmethod
    def from_splits(
        cls,
        data_config,
        train_seqs: list,
        val_seqs: list,
        test_seqs: list | None = None,
        model_preset: str = "",
    ) -> "STPPDataModule":
        """Construct a DataModule from pre-split sequences and a DataConfig.

        Dispatches on ``data_config.protocol``:
        - ``"standard"``: passes the three splits directly to the data module.
        - ``"sliding_window"``: concatenates all splits into a single
          ``raw_seq`` and lets the data module build sliding windows + internal
          train/val/test split.
        """
        import numpy as np

        cfg = data_config
        if cfg.protocol == "sliding_window":
            all_seqs = list(train_seqs) + list(val_seqs) + (list(test_seqs) if test_seqs else [])
            raw_seq = {
                "times": np.concatenate([np.asarray(s["times"]) for s in all_seqs]),
                "locations": np.concatenate([np.asarray(s["locations"]) for s in all_seqs]),
            }
            return cls(
                train_seqs=[],
                val_seqs=[],
                test_seqs=None,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                normalize=cfg.normalize,
                seed=cfg.seed,
                protocol="sliding_window",
                raw_seq=raw_seq,
                paper_lookback=cfg.paper_lookback or 10,
                paper_lookahead=cfg.paper_lookahead,
                paper_split_ratio=cfg.paper_split_ratio,
                model_preset=model_preset,
            )
        else:
            return cls(
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                test_seqs=test_seqs,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                normalize=cfg.normalize,
                seed=cfg.seed,
                protocol="standard",
                model_preset=model_preset,
            )

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_fn,
            num_workers=self.num_workers,
            generator=self._train_generator,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        if self._test_dataset is None:
            return None
        return DataLoader(
            self._test_dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_original_sequence(self, split: str = "val", idx: int = 0) -> dict:
        """Return sequence times and locations in original (un-normalized) space.

        Args:
            split : "train" | "val" | "test"
            idx   : sequence index within the split

        Returns:
            {"times": np.ndarray (L,), "locations": np.ndarray (L, d)}
        """
        dataset = {"train": self._train_dataset, "val": self._val_dataset,
                   "test": self._test_dataset}[split]
        if dataset is None:
            raise ValueError(f"Split {split!r} is not available (dataset is None).")
        seq = dataset.sequences[idx]
        return {
            "times": np.asarray(seq["times"], dtype=np.float64),
            "locations": np.asarray(seq["locations"], dtype=np.float64),
        }
