"""Upstream-style interactive Plotly rendering for intensity cubes."""

from __future__ import annotations

from typing import Any

import numpy as np


def frame_args(duration: int | float) -> dict[str, Any]:
    """Plotly animation settings matching the upstream notebook controls."""
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration},
    }


def _inverse_axis(axis: np.ndarray, scaler, feature_idx: int) -> np.ndarray:
    temp = np.zeros((len(axis), 3), dtype=np.float32)
    temp[:, feature_idx] = axis
    return scaler.inverse_transform(temp)[:, feature_idx].astype(np.float32)


def _maybe_inverse_ranges(
    x_range: np.ndarray,
    y_range: np.ndarray,
    t_range: np.ndarray,
    scaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scaler is None:
        return x_range, y_range, t_range
    return (
        _inverse_axis(x_range, scaler, 0),
        _inverse_axis(y_range, scaler, 1),
        _inverse_axis(t_range, scaler, 2),
    )


def _coerce_lambs(lambs) -> tuple[np.ndarray, int]:
    lambs_arr = np.asarray(lambs, dtype=np.float32)
    if lambs_arr.ndim == 3:
        return lambs_arr, 1
    if lambs_arr.ndim == 4:
        return lambs_arr.transpose(1, 0, 2, 3), int(lambs_arr.shape[0])
    raise ValueError(
        "lambs must have shape (T, X, Y) or (N, T, X, Y), "
        f"got {lambs_arr.shape}."
    )


def _make_trace(
    *,
    go,
    values: np.ndarray,
    x_range: np.ndarray,
    y_range: np.ndarray,
    cmin: float,
    cmax: float,
    heatmap: bool,
    colorscale,
    cauto: bool,
):
    if heatmap:
        return go.Heatmap(
            z=values,
            x=x_range,
            y=y_range,
            zmin=cmin,
            zmax=cmax,
            colorscale=colorscale,
            cauto=cauto,
        )
    return go.Surface(
        z=values,
        x=x_range,
        y=y_range,
        cmin=cmin,
        cmax=cmax,
        colorscale=colorscale,
        cauto=cauto,
    )


def plot_lambst_interactive(
    lambs,
    x_range,
    y_range,
    t_range,
    cmin=None,
    cmax=None,
    scaler=None,
    heatmap: bool = False,
    colorscale="Viridis",
    show: bool = True,
    cauto: bool = False,
    master_title: str = "Spatio-temporal Conditional Intensity",
    subplot_titles=None,
):
    """Render a notebook-style interactive 3D intensity surface animation.

    Parameters match the active upstream notebook API closely enough to support
    direct ``calc_lamb(...) -> plot_lambst_interactive(...)`` usage.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_lambst_interactive(). "
            "Install it in the active environment to render notebook-style HTML."
        ) from exc

    x_range = np.asarray(x_range, dtype=np.float32).reshape(-1)
    y_range = np.asarray(y_range, dtype=np.float32).reshape(-1)
    t_range = np.asarray(t_range, dtype=np.float32).reshape(-1)
    x_range, y_range, t_range = _maybe_inverse_ranges(x_range, y_range, t_range, scaler)
    lambs_arr, n_subplot = _coerce_lambs(lambs)

    expected_shape = (len(t_range), len(x_range), len(y_range))
    if n_subplot == 1:
        actual_shape = tuple(lambs_arr.shape)
    else:
        actual_shape = (int(lambs_arr.shape[0]), int(lambs_arr.shape[2]), int(lambs_arr.shape[3]))
    if actual_shape != expected_shape:
        raise ValueError(
            f"lambs shape {lambs_arr.shape} is incompatible with "
            f"(t, x, y)={expected_shape}."
        )
    if subplot_titles is not None and len(subplot_titles) != n_subplot:
        raise ValueError(
            f"subplot_titles length {len(subplot_titles)} must match "
            f"subplot count {n_subplot}."
        )

    if cmin is None:
        cmin = float(min(0.0, float(lambs_arr.min())))
    if cmax is None:
        cmax = float(lambs_arr.max())

    frames = []
    for i, t_val in enumerate(t_range):
        if n_subplot == 1:
            data = [
                _make_trace(
                    go=go,
                    values=lambs_arr[i],
                    x_range=x_range,
                    y_range=y_range,
                    cmin=cmin,
                    cmax=cmax,
                    heatmap=heatmap,
                    colorscale=colorscale,
                    cauto=cauto,
                )
            ]
        else:
            data = [
                _make_trace(
                    go=go,
                    values=lambs_arr[i, j],
                    x_range=x_range,
                    y_range=y_range,
                    cmin=cmin,
                    cmax=cmax,
                    heatmap=heatmap,
                    colorscale=colorscale,
                    cauto=cauto,
                )
                for j in range(n_subplot)
            ]
        frames.append(go.Frame(data=data, name=f"{float(t_val):.2f}"))

    if n_subplot == 1:
        fig = go.Figure(frames=frames)
        initial = [_make_trace(
            go=go,
            values=lambs_arr[0],
            x_range=x_range,
            y_range=y_range,
            cmin=cmin,
            cmax=cmax,
            heatmap=heatmap,
            colorscale=colorscale,
            cauto=cauto,
        )]
        fig.add_traces(initial)
    else:
        specs = [[{"type": "xy" if heatmap else "scene"} for _ in range(n_subplot)]]
        fig = make_subplots(
            rows=1,
            cols=n_subplot,
            horizontal_spacing=0.05,
            specs=specs,
            subplot_titles=subplot_titles,
        )
        for j in range(n_subplot):
            fig.add_trace(
                _make_trace(
                    go=go,
                    values=lambs_arr[0, j],
                    x_range=x_range,
                    y_range=y_range,
                    cmin=cmin,
                    cmax=cmax,
                    heatmap=heatmap,
                    colorscale=colorscale,
                    cauto=cauto,
                ),
                row=1,
                col=j + 1,
            )
        fig.frames = frames

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0.0,
            "steps": [
                {
                    "args": [[frame.name], frame_args(0)],
                    "label": frame.name,
                    "method": "animate",
                }
                for frame in frames
            ],
        }
    ]

    if heatmap:
        fig.update_xaxes(title_text="x")
        fig.update_yaxes(title_text="y")
    else:
        fig.update_scenes(
            aspectmode="cube",
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="intensity",
            zaxis=dict(range=[cmin, cmax], autorange=False),
        )

    fig.update_layout(
        title=master_title,
        width=500 * n_subplot + 180,
        height=700,
        updatemenus=[
            {
                "buttons": [
                    {"args": [None, frame_args(1)], "label": "Play", "method": "animate"},
                    {"args": [[None], frame_args(0)], "label": "Pause", "method": "animate"},
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0.0,
            }
        ],
        sliders=sliders,
    )
    if show:
        fig.show()
    return fig


__all__ = ["frame_args", "plot_lambst_interactive"]
