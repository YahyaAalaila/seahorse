"""Shared rendering helpers for the replacement evaluation stack."""

from __future__ import annotations

from pathlib import Path


def plot_styles(plot_style: str) -> list[str]:
    if plot_style == "both":
        return ["2d", "3d"]
    return [plot_style]


def write_gif_if_requested(
    png_paths: list[str],
    *,
    out_path: Path,
    fps: float,
) -> str | None:
    if not png_paths:
        return None
    try:
        import imageio.v2 as imageio
    except Exception:
        return None
    images = [imageio.imread(path) for path in png_paths]
    duration = 1.0 / max(float(fps), 1e-6)
    imageio.mimsave(out_path, images, duration=duration)
    return str(out_path)
