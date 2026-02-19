"""
PyTorch Dataset for spatiotemporal point process sequences.
Handles variable-length sequences with padding and collation.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import numpy as np


class STPPDataset(Dataset):
    """
    Dataset for STPP sequences.
    
    Each sample is a dict with:
        - times: (N,) float32
        - locations: (N, d) float32
        - event_covariates: (N, p) float32  [optional]
        - field_covariates: (N, r) float32  [optional, pre-evaluated at events]
    """

    def __init__(
        self,
        sequences: List[Dict],
        normalize_time: bool = True,
        normalize_space: bool = True,
        min_length: int = 3,
        cov_mean: Optional[np.ndarray] = None,
        cov_std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            sequences: list of sequence dicts (times, locations, field_covariates, …)
            normalize_time: z-score normalise event times
            normalize_space: z-score normalise event locations
            min_length: drop sequences shorter than this
            cov_mean: if provided, use this mean to normalise field_covariates
                      instead of computing it from this split's data.
                      Pass train_dataset.cov_mean when creating val/test datasets
                      to keep normalisation consistent across splits.
            cov_std:  matching std array (same contract as cov_mean)
        """
        # Filter short sequences
        self.sequences = [s for s in sequences if len(s["times"]) >= min_length]

        # Compute normalization stats for times and locations
        all_times = np.concatenate([s["times"] for s in self.sequences])
        all_locs = np.concatenate([s["locations"] for s in self.sequences])

        self.time_mean = all_times.mean() if normalize_time else 0.0
        self.time_std = all_times.std() + 1e-8 if normalize_time else 1.0
        self.loc_mean = all_locs.mean(axis=0) if normalize_space else np.zeros(all_locs.shape[1])
        self.loc_std = all_locs.std(axis=0) + 1e-8 if normalize_space else np.ones(all_locs.shape[1])

        self.normalize_time = normalize_time
        self.normalize_space = normalize_space

        # Field covariate normalization stats.
        # If caller supplies external stats (e.g. from the training split) use
        # those; otherwise compute from this split's data so that the dataset
        # is self-contained when used standalone.
        if cov_mean is not None and cov_std is not None:
            self.cov_mean = np.asarray(cov_mean, dtype=np.float32)
            self.cov_std  = np.asarray(cov_std,  dtype=np.float32)
        else:
            cov_arrays = [
                s["field_covariates"] for s in self.sequences
                if "field_covariates" in s
                and s["field_covariates"] is not None
                and len(s["field_covariates"]) > 0
            ]
            if cov_arrays:
                all_covs = np.concatenate(cov_arrays, axis=0)
                self.cov_mean = all_covs.mean(axis=0).astype(np.float32)
                self.cov_std  = (all_covs.std(axis=0) + 1e-8).astype(np.float32)
            else:
                self.cov_mean = None
                self.cov_std  = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        times = seq["times"].copy()
        locs = seq["locations"].copy()

        if self.normalize_time:
            times = (times - self.time_mean) / self.time_std
        if self.normalize_space:
            locs = (locs - self.loc_mean) / self.loc_std

        item = {
            "times": torch.tensor(times, dtype=torch.float32),
            "locations": torch.tensor(locs, dtype=torch.float32),
            "length": len(times),
        }

        if "event_covariates" in seq and seq["event_covariates"] is not None:
            item["event_covariates"] = torch.tensor(
                seq["event_covariates"], dtype=torch.float32
            )

        if "field_covariates" in seq and seq["field_covariates"] is not None:
            cov = seq["field_covariates"].copy()
            if self.cov_mean is not None and len(cov) > 0:
                cov = (cov - self.cov_mean) / self.cov_std
            item["field_covariates"] = torch.tensor(cov, dtype=torch.float32)

        if "marks" in seq and seq["marks"] is not None:
            item["marks"] = torch.tensor(seq["marks"], dtype=torch.long)

        return item


def collate_fn(batch: List[Dict]) -> Dict[str, Tensor]:
    """
    Collate variable-length STPP sequences with padding.
    
    Returns:
        times: (B, N_max) — padded times
        locations: (B, N_max, d) — padded locations
        lengths: (B,) — actual lengths
        event_covariates: (B, N_max, p) or None
        field_covariates: (B, N_max, r) or None
    """
    lengths = torch.tensor([item["length"] for item in batch])
    N_max = lengths.max().item()
    B = len(batch)
    d = batch[0]["locations"].shape[-1]

    times = torch.zeros(B, N_max)
    locations = torch.zeros(B, N_max, d)

    has_event_cov = "event_covariates" in batch[0]
    has_field_cov = "field_covariates" in batch[0]
    has_marks = "marks" in batch[0]

    event_covariates = None
    field_covariates = None
    marks_out = None
    if has_event_cov:
        p = batch[0]["event_covariates"].shape[-1]
        event_covariates = torch.zeros(B, N_max, p)
    if has_field_cov:
        r = batch[0]["field_covariates"].shape[-1]
        field_covariates = torch.zeros(B, N_max, r)
    if has_marks:
        marks_out = torch.zeros(B, N_max, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["length"]
        times[i, :n] = item["times"]
        locations[i, :n] = item["locations"]
        if has_event_cov:
            event_covariates[i, :n] = item["event_covariates"]
        if has_field_cov and item["field_covariates"].shape[0] > 0:
            n_cov = min(n, item["field_covariates"].shape[0])
            field_covariates[i, :n_cov] = item["field_covariates"][:n_cov]
        if has_marks:
            marks_out[i, :n] = item["marks"]

    return {
        "times": times,
        "locations": locations,
        "lengths": lengths,
        "event_covariates": event_covariates,
        "field_covariates": field_covariates,
        "marks": marks_out,
    }
