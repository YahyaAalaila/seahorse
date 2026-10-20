"""Shared machinery for every Hawkes process in this package.

:class:`HawkesProcess` owns the bookkeeping that is identical everywhere — the
random stream, the event record, the simulated-event counter and the canonical
:meth:`simulate` entry point.

:class:`TemporalHawkesProcess` adds the purely temporal Ogata thinning loop.
Concrete classes supply two hooks, :meth:`~TemporalHawkesProcess._conditional_intensity`
and :meth:`~TemporalHawkesProcess._upper_bound`, and inherit both the simulator
and the intensity accessor. Defining the accessor in terms of the *same*
function the simulator thins against is what keeps the two from diverging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = [
    "HawkesProcess",
    "MarkedTemporalHawkesProcess",
    "MultivariateTemporalHawkesProcess",
    "TemporalHawkesProcess",
]

#: Anything :func:`numpy.random.default_rng` accepts.
SeedLike = int | np.random.Generator | np.random.SeedSequence | None

#: Slots a fresh buffer reserves. Small enough to be free, large enough that a
#: short run never reallocates.
_MIN_CAPACITY = 8


class _EventBuffer:
    """Amortised-growth storage for the event record.

    Through 0.3.x each accepted event did ``Events = np.append(Events, event)``,
    which allocates a new array and copies the whole record every time: to write
    down ``n`` events it moved ``O(n^2)`` bytes. Doubling makes that ``O(n)``
    amortised.

    The public face is a **view** onto the filled prefix, never a copy, and that
    is safe for a reason worth stating. A view of length ``n`` is a stable
    snapshot of the first ``n`` events: growth allocates a new buffer and leaves
    the old one — and every view onto it — holding exactly what it held before,
    while an in-place write at index ``n`` cannot reach a view that stops at
    ``n``. So ``a = p.events; p.simulate(1)`` leaves ``a`` reading the same
    events it read before, which is what ``np.append`` did too.

    One buffer serves both layouts because the event count is the **last** axis
    in each: ``(n,)`` for a temporal record, ``(ndim + 1, n)`` for a
    spatio-temporal one, whose first row holds the times. Everything here is
    therefore indexed ``[..., :count]``.

    Parameters
    ----------
    rows : int or None
        Number of rows in the record, or ``None`` for a flat one.

    layout : str, optional
        What the rows after row 0 hold, used only to phrase the error
        :meth:`replace` raises. A multivariate record's rows are not all
        coordinates, so the default wording would misdescribe it.

        .. versionadded:: 1.0.0

    .. versionadded:: 0.4.0
    """

    __slots__ = ("_count", "_data", "_layout", "_rows")

    def __init__(self, rows: int | None = None, *, layout: str = "of coordinates") -> None:
        self._rows = rows
        self._layout = layout
        self._count = 0
        self._data = np.empty(self._shaped(0), dtype=float)

    def _shaped(self, capacity: int) -> tuple[int, ...]:
        """Return the buffer's shape at a given capacity."""
        return (capacity,) if self._rows is None else (self._rows, capacity)

    @property
    def view(self) -> np.ndarray:
        """The filled prefix, as a view.

        Empty until the first event is accepted, and holding only real events at
        every moment of the object's life. Before 0.2.0 a fictitious event sat
        at ``t = 0`` until the first :meth:`~HawkesProcess.simulate` call
        finished, contributing to every intensity sum in the meantime -- so the
        first events were drawn from a process with one phantom excitation, and
        the event they were conditioned on was then deleted from the record.
        """
        return self._data[..., : self._count]

    def append(self, event: Any) -> None:
        """Record one event: a scalar time, or a column of time and coordinates."""
        if self._count == self._data.shape[-1]:
            self._reserve(max(_MIN_CAPACITY, 2 * self._count))
        self._data[..., self._count] = event
        self._count += 1

    @property
    def last_time(self) -> float:
        """Time of the most recent event, or ``0.0`` when the record is empty.

        Both layouts put the event count last, so the most recent event is the
        last column; in the spatio-temporal layout its time is row 0. Written
        once here because the two thinning loops each spelled it out inline and
        :meth:`~HawkesProcess.simulate_until` needs a third copy.
        """
        if self._count == 0:
            return 0.0
        if self._rows is None:
            return float(self._data[self._count - 1])
        return float(self._data[0, self._count - 1])

    def _reserve(self, capacity: int) -> None:
        """Move the record into a buffer of the given capacity."""
        bigger = np.empty(self._shaped(capacity), dtype=float)
        bigger[..., : self._count] = self._data[..., : self._count]
        self._data = bigger

    def replace(self, events: Any) -> None:
        """Adopt `events` as the whole record.

        This is what stands behind ``process.events = history``, which is how a
        caller conditions a realisation on events it did not simulate.
        """
        record = np.array(events, dtype=float, copy=True)
        wanted = 1 if self._rows is None else 2
        if record.ndim != wanted:
            raise ValueError(
                f"the event record must be {wanted}-dimensional, got shape {record.shape}"
            )
        if self._rows is not None and record.shape[0] != self._rows:
            raise ValueError(
                f"the event record must have {self._rows} rows -- one of times above "
                f"{self._rows - 1} {self._layout} -- got shape {record.shape}"
            )
        self._data = record
        self._count = record.shape[-1]


def _stalled_message(t: float, bound: float, progress: str) -> str:
    """Explain an exploding process instead of spinning forever.

    When the intensity diverges, the inter-arrival time underflows to zero and
    the thinning loop stops advancing: events pile up at one instant and the
    simulation never returns. Detecting that and raising turns a hang into an
    actionable error.

    `progress` says how far the caller had got in whatever terms its stopping
    rule counts in -- events for :meth:`~HawkesProcess.simulate`, elapsed time
    for :meth:`~HawkesProcess.simulate_until`.

    .. versionchanged:: 0.5.0
       Takes a phrase rather than an accepted/requested pair, because a run
       stopped by a time horizon has no requested count to report.
    """
    return (
        f"Simulation stalled at t={t!r} {progress}: the thinning bound has "
        f"reached {bound!r} and the inter-arrival time has underflowed to zero, "
        "so time can no longer advance. The process is exploding -- the expected "
        "number of offspring per event is at or above one. Reduce the kernel's "
        "mass or use a more slowly growing nonlinearity."
    )


class HawkesProcess(ABC):
    """Base class for all Hawkes processes.

    Parameters
    ----------
    rng : None, int or numpy.random.Generator
        Source of randomness. Pass an ``int`` for a reproducible run, or an
        existing :class:`~numpy.random.Generator` to share one stream with
        other components. ``None`` draws from fresh OS entropy.

        .. versionadded:: 0.2.0
           Simulations are no longer affected by :func:`numpy.random.seed`.

    rows : int or None
        Rows in the event record: ``None`` for a bare sequence of times,
        ``ndim + 1`` when each event also carries a location.

        .. versionadded:: 0.4.0

    Attributes
    ----------
    events : numpy.ndarray
        Simulated events. Shape ``(n,)`` for temporal processes and
        ``(ndim + 1, n)`` for spatio-temporal ones, where row 0 holds times.
        Assignable: setting it conditions the realisation on a history the
        process did not simulate.

        .. versionchanged:: 0.4.0
           Renamed from ``Events``, and a view onto a growing buffer rather than
           a fresh array per event — see :class:`_EventBuffer` for why that is
           the same thing to a reader.

        .. versionchanged:: 0.5.0
           The ``Events`` alias is gone.
    n_simulated : int
        Number of events accepted so far across all :meth:`simulate` calls.

        .. versionchanged:: 0.4.0
           Renamed from ``Sim_num``.

        .. versionchanged:: 0.5.0
           The ``Sim_num`` alias is gone.
    rng : numpy.random.Generator
        The stream every draw is taken from.
    """

    def __init__(
        self,
        *,
        rng: SeedLike = None,
        rows: int | None = None,
        layout: str = "of coordinates",
    ) -> None:
        self.rng = np.random.default_rng(rng)
        self.n_simulated = 0
        self._events = _EventBuffer(rows, layout=layout)

    @property
    def events(self) -> np.ndarray:
        """The realised events, as a view onto the record."""
        return self._events.view

    @events.setter
    def events(self, value: Any) -> None:
        self._events.replace(value)
        self._record_replaced()

    def _record_replaced(self) -> None:  # noqa: B027 - a hook, not an abstract method
        """React to the record being replaced wholesale rather than appended to.

        Deliberately not abstract: a class that derives nothing from the record
        has nothing to do here, and making every process implement an empty
        method to say so is how a hook becomes noise.
        :class:`~hawkes_package.exponential.ExponentialHawkes` is the one class
        that overrides it -- it carries a decaying sum, and a replaced record
        with the same length and last time would otherwise be described by a sum
        belonging to the record that is gone.
        """

    @abstractmethod
    def _propagate(self, k: int) -> None:
        """Simulate exactly `k` further events, appending them to :attr:`events`."""

    @abstractmethod
    def _propagate_until(self, t_end: float, start: float) -> None:
        """Simulate the events falling in ``(start, t_end]``, appending them.

        `start` is where the thinning loop resumes, which is not always the last
        recorded event time -- see :meth:`simulate_until`.
        """

    def simulate(self, k: int) -> None:
        """Simulate `k` further events and append them to :attr:`events`.

        Calling this repeatedly continues the same realisation; it does not
        restart it.

        Parameters
        ----------
        k : int
            Number of events to generate. ``0`` is a no-op.

        Raises
        ------
        ValueError
            If `k` is negative.
        """
        k_int = int(k)
        if k_int != k:
            raise ValueError(f"k must be a whole number of events, got {k!r}")
        if k_int < 0:
            raise ValueError(f"k must be non-negative, got {k_int}")
        if k_int == 0:
            return
        self._propagate(k_int)

    def simulate_until(self, t_end: float, *, start: float | None = None) -> None:
        """Simulate every event up to time `t_end` and append them to :attr:`events`.

        The complement of :meth:`simulate`: that one fixes the number of events
        and lets the horizon fall where it may, this one fixes the horizon and
        lets the count fall where it may. Which is the one a *forecast* needs,
        because "how many events by time T" is the question being asked, and
        because simulating a fixed count and truncating cannot express the
        outcome "no events at all in ``(start, t_end]``" -- the most informative
        one there is.

        Parameters
        ----------
        t_end : float
            End of the horizon, inclusive. ``t_end <= start`` is a no-op.
        start : float, optional
            Where the loop resumes. Defaults to the last recorded event time, or
            ``0.0`` when the record is empty -- which is what continues the
            realisation, exactly as :meth:`simulate` does.

            Passing a `start` **later** than the last recorded event conditions
            on more than the record holds: it asserts that nothing happened in
            ``(last event, start]``. That is the forecasting case, and it is the
            reason this argument exists.

        Raises
        ------
        ValueError
            If `t_end` or `start` is not finite, or if `start` lies before the
            last recorded event. Rewinding would leave the record holding events
            the loop is about to generate around, so the result would be neither
            the conditional process nor a fresh one.

        Notes
        -----
        The first candidate past `t_end` is discarded rather than appended, and
        the acceptance test is never reached for it. Truncating the thinning
        loop this way is exact rather than approximate: a candidate beyond the
        horizon cannot influence whether an earlier one was accepted, and the
        exponential clock is memoryless, so the events on ``(start, t_end]`` are
        distributed exactly as the untruncated process would have put them
        there. The variate that carried the loop past `t_end` is consumed all
        the same, so a seeded ``simulate_until`` and a seeded ``simulate`` do
        not leave the stream in the same place.

        Examples
        --------
        >>> process = ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=0)
        >>> process.simulate_until(10.0)
        >>> bool(process.events.max() <= 10.0)
        True

        .. versionadded:: 0.5.0
        """
        horizon = float(t_end)
        if not np.isfinite(horizon):
            raise ValueError(f"t_end must be finite, got {t_end!r}")

        last = self._events.last_time
        if start is None:
            origin = last
        else:
            origin = float(start)
            if not np.isfinite(origin):
                raise ValueError(f"start must be finite, got {start!r}")
            if origin < last:
                raise ValueError(
                    f"start={origin!r} lies before the last recorded event at {last!r}; "
                    "a realisation cannot be rewound. Pass a start at or after the last "
                    "event, or simulate a fresh process."
                )

        if horizon <= origin:
            return
        self._propagate_until(horizon, origin)


class _IntensityCursor:
    """How the thinning loop reads the intensity as it walks forward.

    One per :meth:`TemporalHawkesProcess._propagate` call. This default asks the
    two hooks at the cursor's current time, which is exactly what the loop did
    inline before the cursor existed -- so every class that does not override
    :meth:`TemporalHawkesProcess._cursor` draws the same realisation the inline
    loop drew, down to the order the hooks are called in.

    It exists so that a class whose kernel admits a recursion can carry state
    between steps *without* a second copy of the loop. Two copies of a thinning
    loop that must consume the same random stream in the same order is the kind
    of duplication that drifts, and there are already three of them in this
    module for the three record layouts.

    .. versionadded:: 1.0.0
    """

    __slots__ = ("process", "time")

    def __init__(self, process: TemporalHawkesProcess, start: float) -> None:
        self.process = process
        self.time = float(start)

    def move_to(self, t: float) -> None:
        """Walk to `t`, which the loop guarantees is strictly later."""
        self.time = t

    def bound(self) -> float:
        """Return ``M`` at the current time, refusing a non-positive one."""
        return self.process._bound_at(self.time)

    def intensity(self) -> float:
        """Return ``lambda(t | H_t)`` at the current time."""
        return self.process._conditional_intensity(self.time)

    def accept(self) -> None:
        """Absorb the event the loop has just recorded at the current time."""


class TemporalHawkesProcess(HawkesProcess):
    """A Hawkes process on the time line, simulated by Ogata's thinning.

    Subclasses implement :meth:`_conditional_intensity` and :meth:`_upper_bound`.
    """

    @abstractmethod
    def _conditional_intensity(self, t: float) -> float:
        """Conditional intensity lambda(t | H_t), strictly excluding events at `t`."""

    @abstractmethod
    def _upper_bound(self, t: float) -> float:
        """Return a value M >= lambda(s | H_s) for all s >= `t` until the next event.

        This is the ``M`` of Ogata's algorithm. It must dominate the intensity
        over the whole interval the next candidate can land in, or the thinning
        silently degenerates towards a Poisson process.
        """

    def _require_positive_bound(self, bound: float, t: float) -> float:
        """Return `bound`, refusing a non-positive one.

        Shared by every cursor rather than living inside :meth:`_bound_at`, so
        that a cursor computing the bound its own way still fails the same way.
        """
        if not bound > 0:
            raise RuntimeError(
                f"Non-positive thinning bound M={bound!r} at t={t!r}; the "
                "kernel or nonlinearity must keep the intensity positive."
            )
        return bound

    def _bound_at(self, t: float) -> float:
        """Return the thinning bound at `t`, refusing a non-positive one."""
        return self._require_positive_bound(self._upper_bound(t), t)

    def _cursor(self, start: float) -> _IntensityCursor:
        """Return the object the thinning loop reads the intensity through.

        The default asks the two hooks at the cursor's current time, which is
        what the loop did inline before the cursor existed -- so overriding
        *nothing* leaves a class bit-identical.
        :class:`~hawkes_package.exponential.ExponentialHawkes`
        overrides it with a cursor that carries the sum forward instead of
        rebuilding it, which is exact only because its kernel is memoryless.
        """
        return _IntensityCursor(self, start)

    def _propagate(self, k: int) -> None:
        # The bootstrap time is a local, not an element of `events`.
        cursor = self._cursor(self._events.last_time)
        accepted = 0

        while accepted < k:
            t = cursor.time
            bound = cursor.bound()
            advanced = t + self.rng.exponential() / bound
            if not advanced > t:
                raise RuntimeError(
                    _stalled_message(t, bound, f"after {accepted} of {k} requested events")
                )
            cursor.move_to(advanced)

            if self.rng.uniform() * bound <= cursor.intensity():
                self._events.append(advanced)
                cursor.accept()
                accepted += 1
                # Counted per event, so `n_simulated == len(events)` still holds
                # if the loop raises partway through.
                self.n_simulated += 1

        self._save_cursor(cursor)

    def _propagate_until(self, t_end: float, start: float) -> None:
        cursor = self._cursor(start)
        accepted = 0

        while True:
            t = cursor.time
            bound = cursor.bound()
            advanced = t + self.rng.exponential() / bound
            if not advanced > t:
                raise RuntimeError(
                    _stalled_message(t, bound, f"after {accepted} events past t={start!r}")
                )

            # Tested before the acceptance test rather than after it, so the
            # candidate that ends the run costs one intensity evaluation fewer
            # rather than one more.
            if advanced > t_end:
                self._save_cursor(cursor)
                return

            cursor.move_to(advanced)
            if self.rng.uniform() * bound <= cursor.intensity():
                self._events.append(advanced)
                cursor.accept()
                accepted += 1
                self.n_simulated += 1

    def _save_cursor(self, cursor: _IntensityCursor) -> None:
        """Keep `cursor` for the next call, if the class has state worth keeping.

        A no-op by default. It exists for
        :meth:`~hawkes_package.exponential.ExponentialHawkes._cursor`, whose
        carried sum must survive between calls: rebuilding it from the record
        gives a different value in the last bits than the one the uninterrupted
        loop held, and ``simulate(1); simulate(1)`` is asserted to be
        ``simulate(2)`` bit for bit.
        """

    def intensity_over_interval(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the conditional intensity on `x` merged with the event times.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate. The realised event times are merged in
            and the result is sorted and de-duplicated, so the returned grid is
            generally longer than `x`.

        Returns
        -------
        times : numpy.ndarray
            The sorted evaluation grid.
        intensity : numpy.ndarray
            ``lambda(t | H_t)`` at each entry of `times`, including the
            background rate.

        Notes
        -----
        Because the intensity excludes events at exactly `t`, the value
        returned at an event time is the left limit — the pre-jump value.

        .. versionchanged:: 0.2.0
           The returned values now include the background intensity. Before
           0.2.0 :class:`~hawkes_package.exponential.ExponentialHawkes` omitted
           it, so its curves sat a constant ``mu`` below the intensity the
           simulator actually thinned against.
        """
        times = np.unique(np.append(np.asarray(x, dtype=float).ravel(), self.events))
        intensity = np.array([self._conditional_intensity(float(t)) for t in times])
        return times, intensity


class MarkedTemporalHawkesProcess(TemporalHawkesProcess):
    r"""A Hawkes process whose events carry a mark that scales their excitation.

    .. math::

        \lambda(t \mid H_t) = \varphi\!\left( \mu
            + \sum_{t_i < t} g(m_i)\, \kappa(t - t_i) \right)

    with :math:`g` a non-negative **productivity**: an event with a larger mark
    produces more offspring, in the style of the ETAS magnitude term. Subclasses
    supply :meth:`_productivity`, :meth:`_draw_mark` and the two inherited hooks;
    the loop and the record come from here.

    **An unbounded productivity does not threaten the bound**, which is worth
    stating because it looks as though it should. The intensity sums over
    :math:`t_i < t` strictly and the bound sums over :math:`t_i \le t_0`, so both
    range over a finite set of marks that have *already been observed*. The
    supremum of :math:`g` over the mark distribution never enters: a candidate
    accepted at :math:`t` draws its mark afterwards, by which time the bound for
    that step has done its work.

    What the bound does need is :math:`g \ge 0`, because
    :math:`\sup(af) = a\sup(f)` holds only for non-negative *a* -- the same step
    the non-negative excitation matrix serves in
    :class:`MultivariateTemporalHawkesProcess`.

    **The mark is drawn after acceptance**, and only after. That is what keeps a
    marked run comparable with an unmarked one: until the first mark is drawn the
    two consume the same stream in the same order.

    Parameters
    ----------
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`HawkesProcess`.

    Attributes
    ----------
    events : numpy.ndarray
        Shape ``(2, n)``: row 0 holds times, row 1 the marks.
    marks : numpy.ndarray
        Row 1 of the record.

    .. versionadded:: 1.0.0
    """

    def __init__(self, *, rng: SeedLike = None) -> None:
        super().__init__(rng=rng, rows=2, layout="holding the event mark")

    @abstractmethod
    def _productivity(self, marks: np.ndarray) -> np.ndarray:
        """Return ``g(m)`` for each mark, elementwise and non-negative."""

    @abstractmethod
    def _draw_mark(self) -> float:
        """Draw one mark for an accepted event, from ``self.rng``."""

    @property
    def marks(self) -> np.ndarray:
        """The mark of each recorded event."""
        return np.asarray(self.events[1], dtype=float)

    def _past(self, t: float, inclusive: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return the times and marks of the events before `t`, as a pair of views."""
        record = self.events
        times = record[0]
        keep = times <= t if inclusive else times < t
        return times[keep], record[1][keep]

    def _propagate(self, k: int) -> None:
        t = self._events.last_time
        accepted = 0

        while accepted < k:
            bound = self._bound_at(t)
            advanced = t + self.rng.exponential() / bound
            if not advanced > t:
                raise RuntimeError(
                    _stalled_message(t, bound, f"after {accepted} of {k} requested events")
                )
            t = advanced

            if self.rng.uniform() * bound <= self._conditional_intensity(t):
                # After acceptance, never before: the bound for this step was
                # computed from marks already recorded, and drawing earlier would
                # consume a variate on every rejected candidate as well.
                self._events.append((t, self._draw_mark()))
                accepted += 1
                self.n_simulated += 1

    def _propagate_until(self, t_end: float, start: float) -> None:
        t = start
        accepted = 0

        while True:
            bound = self._bound_at(t)
            advanced = t + self.rng.exponential() / bound
            if not advanced > t:
                raise RuntimeError(
                    _stalled_message(t, bound, f"after {accepted} events past t={start!r}")
                )
            t = advanced

            if t > t_end:
                return

            if self.rng.uniform() * bound <= self._conditional_intensity(t):
                self._events.append((t, self._draw_mark()))
                accepted += 1
                self.n_simulated += 1


class MultivariateTemporalHawkesProcess(TemporalHawkesProcess):
    r"""A Hawkes process on the time line whose events carry one of `n_types` types.

    Type *j* excites type *i*, so the intensity is a vector and the process is
    mutually exciting:

    .. math::

        \lambda_i(t \mid H_t) = \varphi_i\!\left( \mu_i + \sum_j A_{ij}
            \sum_{t_k < t,\; c_k = j} \kappa(t - t_k) \right).

    Subclasses implement :meth:`_component_intensities` and
    :meth:`~TemporalHawkesProcess._upper_bound`; the loop, the accessor and the
    record are inherited.

    **One bound, on the total.** The thinning bound dominates
    :math:`\sum_i \lambda_i`, not each component separately, and the type is
    drawn at acceptance. The domination argument is the univariate one applied
    per component and then summed: the event set does not grow before the next
    accepted event, each term obeys
    :math:`A_{ij}\kappa(s - t_k) \le A_{ij}\sup_{u \ge t - t_k}\kappa(u)`, and a
    sum of finitely many dominating scalars dominates the sum of the dominated
    ones.

    That middle step needs :math:`A_{ij} \ge 0`, because
    :math:`\sup(af) = a\sup(f)` only for non-negative *a*. Concrete subclasses
    must refuse a negative entry at construction; inhibitory cross-excitation is
    not a setting this class can be given, it is a different bound argument.

    **The type costs no extra variate.** One exponential and one uniform per
    candidate, exactly as :class:`TemporalHawkesProcess` draws them: the
    cumulative component intensities partition :math:`(0, M]` into one slice per
    type plus the rejection remainder, and which slice the uniform landed in
    *is* the type. So a one-type process here consumes the same stream as the
    univariate classes and reproduces them event for event.

    Parameters
    ----------
    n_types : int
        Number of event types, at least one.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`HawkesProcess`.

    Attributes
    ----------
    n_types : int
        Number of event types.
    events : numpy.ndarray
        Shape ``(2, n)``: row 0 holds times, row 1 the type index as a float.
        Exact for every type index this class admits, since a float64 carries
        integers below ``2 ** 53`` exactly.
    types : numpy.ndarray
        Row 1 of the record, as integers.

    .. versionadded:: 1.0.0
    """

    def __init__(self, n_types: int, *, rng: SeedLike = None) -> None:
        count = int(n_types)
        if count != n_types or count < 1:
            raise ValueError(f"n_types must be a positive whole number, got {n_types!r}")
        super().__init__(rng=rng, rows=2, layout="holding the event type")
        self.n_types = count

    @abstractmethod
    def _component_intensities(self, t: float) -> np.ndarray:
        """Per-type intensity at `t`, shape ``(n_types,)``.

        Strictly excludes events at `t`, as the scalar hook does.
        """

    def _cumulative_intensities(self, t: float) -> np.ndarray:
        """Return the running total of :meth:`_component_intensities`.

        The last entry is the total intensity.

        One expression serves both the acceptance test and
        :meth:`_conditional_intensity`. That is deliberate: ``cumsum(v)[-1]``
        and ``sum(v)`` are free to disagree in the last bit, and defining the
        accessor as one while the loop thins against the other is precisely how
        :class:`~hawkes_package.exponential.ExponentialHawkes` once came to
        report a curve that was not the curve it simulated from.
        """
        return np.cumsum(self._component_intensities(t))

    def _conditional_intensity(self, t: float) -> float:
        """Return the total intensity across types, which is what the bound dominates."""
        return float(self._cumulative_intensities(t)[-1])

    @property
    def types(self) -> np.ndarray:
        """Type index of each recorded event."""
        return np.asarray(self.events[1], dtype=np.intp)

    def _accepted_type(self, t: float, bound: float) -> int | None:
        """Run the acceptance test at `t`, returning the accepted type or ``None``.

        One uniform, drawn and scaled exactly as the univariate loop draws and
        scales it, so the two consume identical streams.
        """
        cumulative = self._cumulative_intensities(t)
        target = self.rng.uniform() * bound
        if target > cumulative[-1]:
            return None
        # `target <= cumulative[-1]` puts the draw inside some slice, so this
        # never returns `n_types`.
        return int(np.searchsorted(cumulative, target, side="left"))

    def _propagate(self, k: int) -> None:
        t = self._events.last_time
        accepted = 0

        while accepted < k:
            bound = self._bound_at(t)
            advanced = t + self.rng.exponential() / bound
            if not advanced > t:
                raise RuntimeError(
                    _stalled_message(t, bound, f"after {accepted} of {k} requested events")
                )
            t = advanced

            kind = self._accepted_type(t, bound)
            if kind is not None:
                self._events.append((t, float(kind)))
                accepted += 1
                self.n_simulated += 1

    def _propagate_until(self, t_end: float, start: float) -> None:
        t = start
        accepted = 0

        while True:
            bound = self._bound_at(t)
            advanced = t + self.rng.exponential() / bound
            if not advanced > t:
                raise RuntimeError(
                    _stalled_message(t, bound, f"after {accepted} events past t={start!r}")
                )
            t = advanced

            if t > t_end:
                return

            kind = self._accepted_type(t, bound)
            if kind is not None:
                self._events.append((t, float(kind)))
                accepted += 1
                self.n_simulated += 1
