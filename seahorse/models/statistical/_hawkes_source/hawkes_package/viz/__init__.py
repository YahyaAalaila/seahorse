r"""The spatio-temporal intensity as an animated three-dimensional surface.

Renders the surface a process lives on, colours it by
:math:`\lambda(t, x \mid H_t)`, and animates it over time into one
self-contained interactive page: a play button, a frame slider, and a camera
that stays orbitable while it plays.

.. code-block:: python

    import numpy as np
    import hawkes_package as hp
    from hawkes_package.spatio_temporal import FundamentalDomain
    from hawkes_package.viz import animate_intensity

    bottle = FundamentalDomain.klein_bottle(3.0, 5.0)
    process = hp.SpatioTemporalHawkesProcess(
        base=lambda x: 0.3,
        spatial=lambda d: 0.8 * np.exp(-2.0 * d),
        temporal=lambda s: 1.5 * np.exp(-1.0 * s),
        domain=bottle,
        monotone_temporal_kernel=True,
        rng=0,
    )
    process.simulate(25)

    frames = animate_intensity(process, np.linspace(0.0, process.events[0, -1], 40), "klein.html")
    print(frames.summary())

Not a runtime dependency
------------------------

The renderer is plotly, and it is an **optional extra**::

    pip install "the-hawkes-package[viz]"

Nothing outside :mod:`hawkes_package.viz._plotly` names it, and that module
imports it inside a function body, so every module here imports with numpy and
scipy alone -- which ``tests/test_api_surface.py`` asserts. Building and
inspecting the frames therefore needs nothing extra; only writing the page does.

The split is deliberate rather than incidental. :func:`intensity_frames` returns
plain arrays evaluated through the simulator's own hooks, so it is the piece
worth testing and the piece a different backend would be written against;
:mod:`~hawkes_package.viz._plotly` is a translation layer with no arithmetic of
its own.

What the pictures claim, and what they do not
---------------------------------------------

Four surfaces are supported, and they are not all equally honest:

* :class:`~hawkes_package.spatio_temporal.Sphere` is drawn as itself. Exact.
* The **projective plane** is drawn as its double cover, the sphere. The
  covering map is a local isometry, so distances on screen are true, and the
  antipodal symmetry of the colouring is the identification made visible.
* The **flat torus** and the **Klein bottle** are drawn as a donut and a
  figure-8 immersion. Neither is isometric -- neither surface admits an
  isometric embedding in three-space, and the Klein bottle admits no embedding
  at all -- so the geodesic distances driving the intensity are *not* the
  distances a viewer measures. :attr:`SurfaceEmbedding.note` says so per
  surface, and the page's caption carries it.

Hyperbolic surfaces are refused rather than approximated: by Hilbert's theorem
they have no isometric picture in three-space, and they deserve a Poincare-disc
treatment rather than a misleading solid.

.. versionadded:: 0.5.0
"""

from ._embedding import SurfaceEmbedding, embed
from ._field import IntensityFrames, event_opacities, intensity_frames
from ._plotly import animate_intensity, build_figure

__all__ = [
    "IntensityFrames",
    "SurfaceEmbedding",
    "animate_intensity",
    "build_figure",
    "embed",
    "event_opacities",
    "intensity_frames",
]
