"""Dynamic source response ``S(omega)``: frequency response of a source term.

A mass source (a fuel/air injector) or a heat source carries a **dynamic** part: a
fluctuating injection / heat release whose amplitude responds to the unsteady flow
(perturbations) elsewhere in the network.  The classic example is a velocity-driven
flame transfer function (FTF): the heat release fluctuates with the acoustic velocity
``u'`` an edge upstream of the flame, with a gain and a time lag. This feedback is
what makes the perturbation operator non-self-adjoint and drives thermoacoustic
instability.

The general response is a **superposition of transfer functions**, each on its own
reference edge and quantity::

    q'(omega) / q_bar = sum_k  gain_k * F_k(omega) * ( phi'_k(omega) / phi_bar_k )

where ``q'`` is the fluctuation of the modulated source quantity (heat release for a
flame, injected mass-flow for a source), ``q_bar`` its mean, ``F_k`` a (generally
complex) transfer function of frequency, and ``phi_k`` a reference flow quantity
(``u``, ``p``, ``rho``, ``mdot`` or a composition scalar ``Z:<name>``) at a chosen
reference edge.  Most flames are modelled with a single velocity term.

This module owns only the **specification** (the descriptor + the transfer-function
objects); the mean flow ignores it entirely (a constant mean source is acoustically
passive), and the perturbation layer (:mod:`nefes.perturbation.operator.stamps`)
consumes it to stamp the ``S(omega)`` block of the operator.  Nothing here depends on
the perturbation layer.

Frequency convention
--------------------
Every transfer function is a function of **frequency in Hz** (project convention --
graphs and user input use frequency, not angular frequency).  The perturbation
assembler evaluates ``F(omega / 2 pi)``.  For a stability analysis the frequency is
**complex**, so a transfer function must be analytically continuable
(:attr:`TransferFunction.analytic`); the closed-form models are, a table interpolated
on a real grid is not.  Use the table for the forced response, and for stability work
convert it: :func:`fit_impulse_response` (the recommended route -- a response that dies
out in finite time continues off the real axis with no poles at all) or, for a response
with a genuine resonance, :func:`~nefes.perturbation.continuation.rational_fit`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import List, Optional

import numpy as np

from .parameters import ParamDescriptor
from .parametric import AttributeParams, is_parametric

# multi-term knob addresses: "terms[<k>].<name>"
_TERM_ADDRESS = re.compile(r"^terms\[(\d+)\]\.(.+)$")

# Reference quantities a response term may read at its edge.  Composition scalars are
# named "Z:<scalar-name>" and resolved against the compiled problem's ``scalar_names``.
_QUANTITIES = ("u", "p", "rho", "mdot")

# Which source quantity a descriptor modulates, and the residual it feeds.
_TARGETS = ("Qdot", "mdot")


# ==========================================================================
# Transfer functions  F(f) : (complex) frequency [Hz] -> complex
# ==========================================================================


class TransferFunction:
    """A complex-valued function of frequency ``F(f)`` with ``f`` in Hz.

    Subclasses implement :meth:`__call__`.  Two attributes inform the drivers:

    * :attr:`analytic` -- ``True`` if ``F`` can be evaluated at a *complex* frequency
      (required for the stability eigenproblem, which searches the complex plane).
    * :attr:`max_delay` -- the longest pure time lag [s] the function carries; the
      stability driver uses it to clamp the search contour so ``e^{-i omega tau}``
      does not overflow at large growth/decay rates.
    """

    analytic: bool = True
    max_delay: float = 0.0

    def __call__(self, f):  # pragma: no cover - abstract
        raise NotImplementedError

    def plot(self, freqs, **kwargs):
        """Plot magnitude and phase versus frequency (see :func:`nefes.plotting.plot_transfer_function`)."""
        from ..plotting import plot_transfer_function

        return plot_transfer_function(self, freqs, **kwargs)


class Constant(TransferFunction):
    """A frequency-independent (generally complex) response ``F(f) = value``."""

    analytic = True

    def __init__(self, value):
        self.value = complex(value)

    def __call__(self, f):
        return self.value * np.ones_like(np.asarray(f, dtype=np.complex128))

    def __repr__(self):
        return f"Constant({self.value!r})"


class NTau(AttributeParams, TransferFunction):
    """The ``n-tau`` flame model ``F(f) = n * exp(-i * 2 pi f * tau)``.

    The (generally complex) interaction index ``n`` times a pure time lag ``tau``
    [s].  Entire in frequency, so it is usable in the stability eigenproblem.  Under
    the ``e^{+i omega t}`` convention the factor ``exp(-i omega tau)`` is the causal
    delay of the response behind the driving fluctuation (same sign as the duct
    propagation phase).

    Parameters
    ----------
    n : float or complex
        Interaction index; the gain of the model is ``abs(n)``.
    tau : float
        Time lag [s] (``>= 0`` for a causal response).
    """

    analytic = True
    _PARAM_SPEC = (
        ParamDescriptor("n", doc="interaction index (response gain)"),
        ParamDescriptor("tau", unit="s", lo=0.0, doc="time lag"),
    )

    def __init__(self, n, tau):
        self.n = complex(n)
        self.tau = float(tau)
        self.max_delay = abs(float(tau))

    def __call__(self, f):
        f = np.asarray(f, dtype=np.complex128)
        return self.n * np.exp(-2j * np.pi * f * self.tau)

    def __repr__(self):
        return f"NTau(n={self.n!r}, tau={self.tau!r})"


class NTauLowpass(AttributeParams, TransferFunction):
    """The ``n-tau`` flame with a first-order gain roll-off ``F(f) = n e^{-i 2 pi f tau} / (1 + i f / f_c)``.

    The bare :class:`NTau` model has a frequency-independent gain ``n``, which lets a
    lossless duct destabilize an unbounded comb of high-frequency modes -- unphysical.
    A real flame is a **low-pass** responder: its gain rolls off above a cutoff ``f_c``
    (the flame cannot follow forcing faster than its own response time), bounding the
    instability to a finite band.  This is the canonical model for a converged
    (Nyquist) stability count.

    Entire in the unstable (lower-half ``omega``) plane -- the low-pass pole sits at
    ``f = i f_c`` (the *stable* upper half), so it is analytically continuable for the
    eigenproblem (as long as the search region does not reach down to growth
    ``-2 pi f_c``).

    Parameters
    ----------
    n : float or complex
        Low-frequency interaction index; the zero-frequency gain is ``abs(n)``.
    tau : float
        Time lag [s] (``>= 0`` for a causal response).
    fc : float
        Roll-off cutoff frequency [Hz] (``> 0``).
    """

    analytic = True
    _PARAM_SPEC = (
        ParamDescriptor("n", doc="low-frequency interaction index"),
        ParamDescriptor("tau", unit="s", lo=0.0, doc="time lag"),
        ParamDescriptor("fc", unit="Hz", lo=0.0, lo_open=True, doc="gain roll-off cutoff frequency"),
    )

    def __init__(self, n, tau, fc):
        self.n = complex(n)
        self.tau = float(tau)
        self.fc = float(fc)
        if self.fc <= 0.0:
            raise ValueError(f"roll-off cutoff fc must be positive; got {fc}")
        self.max_delay = abs(float(tau))

    def __call__(self, f):
        f = np.asarray(f, dtype=np.complex128)
        return self.n * np.exp(-2j * np.pi * f * self.tau) / (1.0 + 1j * f / self.fc)

    def __repr__(self):
        return f"NTauLowpass(n={self.n!r}, tau={self.tau!r}, fc={self.fc!r})"


class NTauLowpass2(AttributeParams, TransferFunction):
    """The ``n-tau`` flame with a second-order gain roll-off.

    ``F(f) = n e^{-i 2 pi f tau} / (1 - (f/f_c)^2 + 2 i zeta f / f_c)``, the delayed
    response of a damped second-order oscillator.  Where :class:`NTauLowpass` rolls off
    monotonically, this model can *overshoot* below the cutoff (a gain bump for
    ``zeta < 1/sqrt(2)``) before falling off twice as steeply above it, and its phase
    swings through a further half turn.  Measured V-shaped and swirl-stabilized flames
    show exactly that shape, which the first-order form cannot reproduce.

    Entire in the unstable (lower-half ``omega``) plane -- both poles sit at
    ``f = f_c (i zeta +- sqrt(1 - zeta^2))``, in the *upper* half plane for every
    ``zeta > 0``, so the model is analytically continuable for the eigenproblem as long as
    the search region does not reach up to the nearer of them (for ``zeta < 1`` both sit at
    growth ``-2 pi zeta f_c``).

    Parameters
    ----------
    n : float or complex
        Low-frequency interaction index; the zero-frequency gain is ``abs(n)``.
    tau : float
        Time lag [s] (``>= 0`` for a causal response).
    fc : float
        Roll-off cutoff frequency [Hz] (``> 0``).
    zeta : float
        Damping ratio of the second-order roll-off (``> 0``).  Below
        ``1/sqrt(2)`` the gain peaks near ``f_c``; ``zeta >= 1`` is overdamped.

    See Also
    --------
    NTauLowpass : the first-order roll-off, appropriate for conical flames.

    Examples
    --------
    >>> F = n_tau_lowpass2(1.0, 2.0e-3, 200.0, 0.5)
    >>> abs(complex(F(0.0)))  # unit gain at zero frequency
    1.0
    """

    analytic = True
    _PARAM_SPEC = (
        ParamDescriptor("n", doc="low-frequency interaction index"),
        ParamDescriptor("tau", unit="s", lo=0.0, doc="time lag"),
        ParamDescriptor("fc", unit="Hz", lo=0.0, lo_open=True, doc="gain roll-off cutoff frequency"),
        ParamDescriptor("zeta", lo=0.0, lo_open=True, doc="roll-off damping ratio"),
    )

    def __init__(self, n, tau, fc, zeta):
        self.n = complex(n)
        self.tau = float(tau)
        self.fc = float(fc)
        self.zeta = float(zeta)
        if self.fc <= 0.0:
            raise ValueError(f"roll-off cutoff fc must be positive; got {fc}")
        if self.zeta <= 0.0:
            raise ValueError(f"damping ratio zeta must be positive; got {zeta}")
        self.max_delay = abs(float(tau))

    def __call__(self, f):
        f = np.asarray(f, dtype=np.complex128)
        x = f / self.fc  # frequency scaled by the cutoff
        return self.n * np.exp(-2j * np.pi * f * self.tau) / (1.0 - x * x + 2j * self.zeta * x)

    def __repr__(self):
        return f"NTauLowpass2(n={self.n!r}, tau={self.tau!r}, fc={self.fc!r}, zeta={self.zeta!r})"


class FiniteImpulseResponse(AttributeParams, TransferFunction):
    """A flame response given by its impulse response ``h`` sampled every ``dt`` seconds.

    ``F(f) = sum_j h_j e^{-i 2 pi f j dt}``, the frequency response of the discrete impulse
    response ``h_0, h_1, ...``.  This is the form a flame response takes when it is
    identified from a broadband simulation or experiment: the coefficients ``h_j`` are the
    heat-release response to a unit velocity impulse ``j dt`` seconds earlier, and their
    sum is the zero-frequency gain ``F(0)``.

    A finite sum of exponentials is *entire* -- it has no poles anywhere -- so unlike a
    rational fit of the same data it continues into the whole complex-frequency plane and
    is safe to hand to the stability eigenproblem at any growth rate.

    Parameters
    ----------
    h : array_like
        Real impulse-response coefficients ``h_j``, ``j = 0 .. J``, in order of increasing
        lag.  ``sum(h)`` is the zero-frequency gain.
    dt : float
        Sampling interval [s] of the impulse response (``> 0``).  Frequencies above the
        Nyquist limit ``1 / (2 dt)`` are aliased, as for any sampled response.

    Attributes
    ----------
    h, dt : ndarray, float
        The coefficients and their spacing.
    lags : ndarray
        The lag ``j dt`` [s] of each coefficient.

    See Also
    --------
    fit_impulse_response : build this object from tabulated frequency samples.
    NTau : the single-lag limit, ``h`` a lone spike.
    Tabulated : a response held as frequency samples, usable only on the real axis.
    nefes.perturbation.continuation.rational_fit : the alternative for a resonant response.

    Examples
    --------
    A single spike reproduces the ``n-tau`` model:

    >>> import numpy as np
    >>> F = finite_impulse_response([0.0, 0.0, 1.4], 1.0e-3)
    >>> abs(complex(F(0.0)))
    1.4
    >>> bool(np.isclose(complex(F(250.0)), 1.4 * np.exp(-2j * np.pi * 250.0 * 2.0e-3)))
    True
    """

    analytic = True
    # scalar reductions only, never the sampled shape itself: an overall gain multiplier
    # and a bulk shift of every lag
    _PARAM_SPEC = (
        ParamDescriptor("gain", doc="overall multiplier on the identified response"),
        ParamDescriptor("delay", unit="s", lo=0.0, doc="bulk time shift added to every lag"),
    )

    def __init__(self, h, dt, *, gain=1.0, delay=0.0):
        self.h = np.asarray(h, dtype=float).ravel()
        self.dt = float(dt)
        if self.h.size == 0:
            raise ValueError("the impulse response must have at least one coefficient")
        if self.dt <= 0.0:
            raise ValueError(f"the sampling interval dt must be positive; got {dt}")
        self.gain = float(gain)
        self.delay = float(delay)
        if self.delay < 0.0:
            raise ValueError(f"the bulk delay must be non-negative; got {delay}")
        self.lags = np.arange(self.h.size) * self.dt + self.delay
        self.max_delay = float(self.lags[-1])

    def with_value(self, name, value):
        d = self._descriptor(name)
        v = d.validate(value, where=type(self).__name__)
        kw = {"gain": self.gain, "delay": self.delay, name: v}
        return type(self)(self.h, self.dt, **kw)

    @property
    def nyquist(self) -> float:
        """Highest frequency [Hz] the sampled response resolves, ``1 / (2 dt)``."""
        return 0.5 / self.dt

    def __call__(self, f):
        f = np.asarray(f, dtype=np.complex128)
        phase = np.multiply.outer(f, self.lags)  # (..., J+1)
        return self.gain * (self.h * np.exp(-2j * np.pi * phase)).sum(axis=-1)

    def __repr__(self):
        extra = "".join(
            [f", gain={self.gain:g}" if self.gain != 1.0 else "", f", delay={self.delay:g} s" if self.delay else ""]
        )
        return (
            f"FiniteImpulseResponse({self.h.size} coefficients, dt={self.dt!r} s, "
            f"F(0)={self.gain * self.h.sum():.4g}{extra})"
        )


class Tabulated(TransferFunction):
    """A measured transfer function interpolated from a table ``F(freqs) = values``.

    Real-frequency only: this grid interpolant is **not** analytically continuable, so it
    cannot be evaluated at the complex frequencies the stability eigenproblem visits
    (:attr:`analytic` is ``False``).  Use it for the forced response / scattering sweep,
    which stay on the real axis.  For **stability** analysis from the same tabulated data,
    fit it with :func:`fit_impulse_response`: a response that dies out after a finite time
    (a flame, or any compact element without an internal resonator) becomes a
    :class:`FiniteImpulseResponse`, which continues off the real axis exactly and with no
    poles.  Reserve :func:`~nefes.perturbation.continuation.rational_fit` for a response
    with a genuine resonance (a cavity damper, a resonant end plate), whose long ringing a
    finite impulse response would truncate.  A closed-form model (:class:`NTau`) fitted by
    hand works too.

    Magnitude and (unwrapped) phase are interpolated separately so the gain stays
    non-negative and the phase reads smoothly; outside the tabulated band the value
    is held at the nearest endpoint (``extrapolate="hold"``) or set to zero
    (``"zero"``).

    Parameters
    ----------
    freqs : array_like
        Tabulated frequencies [Hz], strictly increasing.
    values : array_like
        Complex transfer-function values at ``freqs``.
    kind : {"linear", "cubic"}, optional
        Interpolation order on the magnitude/phase curves (default ``"cubic"`` when
        SciPy is available and there are enough points, else ``"linear"``).
    extrapolate : {"hold", "zero"}, optional
        Behaviour outside ``[freqs[0], freqs[-1]]`` (default ``"hold"``).
    """

    analytic = False

    def __init__(self, freqs, values, *, kind="cubic", extrapolate="hold"):
        f = np.asarray(freqs, dtype=float)
        v = np.asarray(values, dtype=np.complex128)
        if f.ndim != 1 or v.shape != f.shape:
            raise ValueError("freqs and values must be 1-D arrays of equal length")
        if np.any(np.diff(f) <= 0.0):
            raise ValueError("freqs must be strictly increasing")
        if extrapolate not in ("hold", "zero"):
            raise ValueError("extrapolate must be 'hold' or 'zero'")
        self.freqs = f
        self.values = v
        self.extrapolate = extrapolate
        self._mag = np.abs(v)
        self._phase = np.unwrap(np.angle(v))
        self._kind = kind if (kind == "linear" or f.size >= 4) else "linear"

    def _interp1(self, curve, f):
        if self._kind == "cubic":
            from scipy.interpolate import CubicSpline

            spline = CubicSpline(self.freqs, curve, extrapolate=False)
            out = spline(f)
        else:
            out = np.interp(f, self.freqs, curve, left=np.nan, right=np.nan)
        return out

    def __call__(self, f):
        f = np.asarray(f)
        if np.iscomplexobj(f) and np.any(np.abs(f.imag) > 1e-12 * (np.abs(f.real) + 1.0)):
            raise ValueError(
                "a tabulated transfer function cannot be evaluated at a complex frequency "
                "(real-grid interpolation is not analytic). Use it for the forced response, "
                "or convert it for the stability eigenproblem: fit_impulse_response(freqs, "
                "values, duration=...) for a finite-memory response, rational_fit for a "
                "resonant one, or a closed-form model (e.g. n_tau)."
            )
        fr = np.asarray(f.real if np.iscomplexobj(f) else f, dtype=float)
        mag = self._interp1(self._mag, fr)
        ph = self._interp1(self._phase, fr)
        out = mag * np.exp(1j * ph)
        outside = np.isnan(out)
        if np.any(outside):
            if self.extrapolate == "zero":
                out = np.where(outside, 0.0, out)
            else:  # hold the nearest endpoint value
                held = np.where(fr <= self.freqs[0], self.values[0], self.values[-1])
                out = np.where(outside, held, out)
        return out

    def __repr__(self):
        return f"Tabulated(n={self.freqs.size} points, {self.freqs[0]:.4g}-{self.freqs[-1]:.4g} Hz)"


class _CallableTF(TransferFunction):
    """Wrap a bare ``omega_hz -> complex`` callable as a :class:`TransferFunction`."""

    def __init__(self, fn, *, analytic=False, max_delay=0.0):
        self._fn = fn
        self.analytic = bool(analytic)
        self.max_delay = float(max_delay)

    def __call__(self, f):
        return np.asarray(self._fn(f), dtype=np.complex128)


# -- builders --------------------------------------------------------------


def n_tau(n, tau) -> NTau:
    """The ``n-tau`` flame model ``F(f) = n * exp(-i 2 pi f tau)`` (see :class:`NTau`).

    Parameters
    ----------
    n : float or complex
        Interaction index; the gain of the model is ``abs(n)``.
    tau : float
        Time lag [s] (``>= 0`` for a causal response).

    Returns
    -------
    NTau
    """
    return NTau(n, tau)


def n_tau_lowpass(n, tau, fc) -> NTauLowpass:
    """The ``n-tau`` flame with a first-order gain roll-off (see :class:`NTauLowpass`).

    Parameters
    ----------
    n : float or complex
        Low-frequency interaction index; the zero-frequency gain is ``abs(n)``.
    tau : float
        Time lag [s] (``>= 0`` for a causal response).
    fc : float
        Roll-off cutoff frequency [Hz] (``> 0``).

    Returns
    -------
    NTauLowpass
    """
    return NTauLowpass(n, tau, fc)


def n_tau_lowpass2(n, tau, fc, zeta) -> NTauLowpass2:
    """The ``n-tau`` flame with a second-order gain roll-off (see :class:`NTauLowpass2`).

    Parameters
    ----------
    n : float or complex
        Low-frequency interaction index; the zero-frequency gain is ``abs(n)``.
    tau : float
        Time lag [s] (``>= 0`` for a causal response).
    fc : float
        Roll-off cutoff frequency [Hz] (``> 0``).
    zeta : float
        Damping ratio of the second-order roll-off (``> 0``).

    Returns
    -------
    NTauLowpass2
    """
    return NTauLowpass2(n, tau, fc, zeta)


def constant(value) -> Constant:
    """A frequency-independent (generally complex) response (see :class:`Constant`).

    Parameters
    ----------
    value : float or complex
        The constant response ``F(f) = value``; its gain is ``abs(value)``.

    Returns
    -------
    Constant
    """
    return Constant(value)


def finite_impulse_response(h, dt) -> FiniteImpulseResponse:
    """A response given by its sampled impulse response (see :class:`FiniteImpulseResponse`).

    Parameters
    ----------
    h : array_like
        Real impulse-response coefficients, in order of increasing lag.
    dt : float
        Sampling interval [s] of the impulse response.

    Returns
    -------
    FiniteImpulseResponse

    See Also
    --------
    n_tau : the single-lag limit.
    fit_impulse_response : build the coefficients from tabulated frequency samples.
    """
    return FiniteImpulseResponse(h, dt)


def fit_impulse_response(freqs, values, *, duration, dt=None, smoothing=1.0e-4) -> FiniteImpulseResponse:
    """Fit tabulated frequency-response samples with a finite impulse response.

    This is the recommended bridge from measured (or digitized) frequency-response data
    to the stability eigenproblem.  It assumes the underlying response to a brief
    disturbance dies out within ``duration`` seconds -- true of flames (which forget a
    velocity disturbance after a finite convective time) and of compact elements without
    an internal resonator.  Under that assumption the transfer function is a finite sum
    of pure delays, which evaluates at *any* complex frequency with no poles anywhere,
    so the returned :class:`FiniteImpulseResponse` drops straight into the eigensolver.
    For a response with a genuine resonance (a cavity damper, a resonant end plate),
    whose ringing outlasts any reasonable ``duration``, use
    :func:`~nefes.perturbation.continuation.rational_fit` instead.

    The coefficients solve a least-squares problem on the samples with a smoothness
    penalty on the second difference of the impulse response, so the fit follows the
    trend of the data rather than every measurement wiggle.  By default the sample
    spacing ``dt`` is set from the top of the tabulated band, ``dt = 1 / (2 max f)``,
    so the fit carries no frequency content the data cannot constrain.

    Parameters
    ----------
    freqs : array_like
        Tabulated frequencies [Hz], non-negative, not necessarily uniform.
    values : array_like
        Complex response samples at ``freqs``.
    duration : float
        Memory length [s] of the fitted impulse response; the response is assumed zero
        after this time.  Physically: a few times the largest transport delay in the
        data (readable as the slope of the phase).  Too short truncates the response;
        too long is harmless but leans on the smoothness penalty.
    dt : float, optional
        Sample spacing [s] of the impulse response.  Default ``1 / (2 max(freqs))``,
        which places the resolvable-frequency limit at the edge of the tabulated band.
        A larger ``dt`` cannot represent the top of the band and is rejected.  If the
        response still carries appreciable gain at the top of the band, halve the
        default: exactly at the resolvable limit every basis term turns real, so the
        band edge is matched only loosely at the default spacing, while a finer ``dt``
        restores it (the smoothness penalty governs whatever the data does not
        constrain).
    smoothing : float, optional
        Weight of the second-difference penalty (default ``1e-4``).  Increase for a
        smoother response on noisy data; ``0`` disables the penalty (the problem must
        then be overdetermined by the samples alone).

    Returns
    -------
    FiniteImpulseResponse
        The fitted response, with extra attributes: ``rms_misfit`` and ``max_misfit``
        (root-mean-square and largest absolute deviation from the supplied samples) and
        ``freqs``/``values`` (the samples themselves, so
        :func:`~nefes.plotting.plot_fit` can overlay fit and data).

    Raises
    ------
    ValueError
        On empty/mismatched input, non-finite samples, ``duration`` shorter than one
        sample spacing, or ``dt`` too coarse for the tabulated band.

    See Also
    --------
    FiniteImpulseResponse : the returned object and its evaluation rule.
    Tabulated : keep the raw table for real-axis work (forced response, Nyquist sweep).
    nefes.perturbation.continuation.rational_fit : the alternative for a resonant response.

    Examples
    --------
    Recover a single-lag response from its own samples:

    >>> import numpy as np
    >>> f = np.linspace(0.0, 500.0, 60)
    >>> data = 1.4 * np.exp(-2j * np.pi * f * 2.0e-3)
    >>> F = fit_impulse_response(f, data, duration=5.0e-3, smoothing=0.0)
    >>> bool(abs(complex(F(100.0)) - 1.4 * np.exp(-2j * np.pi * 100.0 * 2.0e-3)) < 1e-6)
    True
    >>> F.rms_misfit < 1e-6
    True
    """
    f = np.asarray(freqs, dtype=float).ravel()
    v = np.asarray(values, dtype=np.complex128).ravel()
    if f.size == 0 or f.shape != v.shape:
        raise ValueError("freqs and values must be non-empty 1-D arrays of equal length")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(v))):
        raise ValueError("freqs and values must be finite")
    if np.any(f < 0.0):
        raise ValueError("freqs must be non-negative")
    f_top = float(f.max())
    if dt is None:
        if f_top <= 0.0:
            raise ValueError("dt cannot be inferred from a zero-frequency-only table; pass dt explicitly")
        dt = 0.5 / f_top
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"the sample spacing dt must be positive; got {dt}")
    if f_top * dt > 0.5 * (1.0 + 1e-9):
        raise ValueError(
            f"dt={dt:g} s cannot represent the tabulated band: its resolvable-frequency limit "
            f"{0.5 / dt:g} Hz lies below the highest tabulated frequency {f_top:g} Hz"
        )
    n = int(round(duration / dt)) + 1
    if n < 2:
        raise ValueError(f"duration={duration!r} s is shorter than one sample spacing dt={dt:g} s")
    smoothing = float(smoothing)
    if smoothing < 0.0:
        raise ValueError(f"smoothing must be non-negative; got {smoothing}")

    lags = np.arange(n) * dt
    basis = np.exp(-2j * np.pi * np.outer(f, lags))
    blocks_lhs = [basis.real, basis.imag]
    blocks_rhs = [v.real, v.imag]
    if smoothing > 0.0 and n >= 3:
        curvature = np.diff(np.eye(n), n=2, axis=0)
        blocks_lhs.append(np.sqrt(smoothing) * curvature)
        blocks_rhs.append(np.zeros(n - 2))
    h, *_ = np.linalg.lstsq(np.vstack(blocks_lhs), np.concatenate(blocks_rhs), rcond=None)

    fitted = FiniteImpulseResponse(h, dt)
    misfit = basis @ h - v
    fitted.rms_misfit = float(np.sqrt(np.mean(np.abs(misfit) ** 2)))
    fitted.max_misfit = float(np.abs(misfit).max())
    fitted.freqs, fitted.values = f, v  # the fitted samples, kept for plot overlays
    return fitted


def tabulated(freqs, values, **kwargs) -> Tabulated:
    """A measured transfer function from a table (see :class:`Tabulated`).

    Parameters
    ----------
    freqs : array_like
        Tabulated frequencies [Hz], strictly increasing.
    values : array_like
        Complex transfer-function values at ``freqs``.
    **kwargs
        Forwarded to :class:`Tabulated` (e.g. ``kind``, ``extrapolate``).

    Returns
    -------
    Tabulated
    """
    return Tabulated(freqs, values, **kwargs)


def as_transfer(obj) -> TransferFunction:
    """Coerce ``obj`` into a :class:`TransferFunction`.

    Parameters
    ----------
    obj : TransferFunction or (n, tau) or number or callable
        An existing :class:`TransferFunction`, an ``(n, tau)`` pair (-> n-tau), a
        real/complex number (-> constant), or a bare ``f -> complex`` callable (wrapped;
        treated as *non*-analytic, so usable only for the forced response unless it is a
        :class:`TransferFunction` declaring ``analytic = True``).

    Returns
    -------
    TransferFunction
    """
    if isinstance(obj, TransferFunction):
        return obj
    if isinstance(obj, (tuple, list)) and len(obj) == 2 and all(np.isscalar(v) for v in obj):
        return NTau(obj[0], obj[1])
    if np.isscalar(obj):
        return Constant(obj)
    if callable(obj):
        return _CallableTF(obj)
    raise TypeError(
        f"cannot interpret {obj!r} as a transfer function; pass a TransferFunction, an "
        "(n, tau) pair, a number, or a callable f->complex"
    )


# ==========================================================================
# Dynamic-source descriptor
# ==========================================================================


@dataclass
class DynamicResponseTerm:
    """One transfer-function term ``gain * F(omega) * (phi'_ref / phi_bar_ref)``.

    Parameters
    ----------
    transfer : TransferFunction or (n, tau) or number or callable
        The frequency response ``F``; coerced via :func:`as_transfer`.
    ref_edge : int
        Edge id whose fluctuation drives this term (e.g. the edge just upstream of a
        flame for a velocity FTF).
    quantity : str, optional
        Reference quantity at ``ref_edge``: ``"u"`` (velocity, default), ``"p"``,
        ``"rho"``, ``"mdot"``, or ``"Z:<name>"`` for a transported composition scalar.
    gain : float, optional
        Real scalar multiplier on this term (default 1.0).  A complex weight belongs in
        ``transfer``, not here: a gain is the magnitude ``abs(F)`` of a response.
    """

    transfer: object
    ref_edge: int
    quantity: str = "u"
    gain: float = 1.0

    def __post_init__(self):
        self.transfer = as_transfer(self.transfer)
        self.ref_edge = int(self.ref_edge)
        q = self.quantity
        if q not in _QUANTITIES and not q.startswith("Z:"):
            raise ValueError(f"quantity must be one of {_QUANTITIES} or 'Z:<scalar-name>'; got {q!r}")
        # a gain is the magnitude abs(F) of a response, hence real; a complex weight is a
        # response in its own right and belongs in ``transfer``
        if isinstance(self.gain, complex):
            raise TypeError(
                f"gain must be real (a gain is abs(F)); pass a complex weight in 'transfer'; got {self.gain!r}"
            )
        self.gain = float(self.gain)

    _GAIN_DESC = ParamDescriptor("gain", doc="real multiplier on this response term")

    def param_descriptors(self):
        """This term's ``gain`` plus the transfer's own knobs (the term's name wins a clash)."""
        descs = [self._GAIN_DESC]
        if is_parametric(self.transfer):
            descs += [d for d in self.transfer.param_descriptors() if d.name != "gain"]
        return tuple(descs)

    def get(self, name):
        """Current value of one knob (own ``gain``, else delegated to the transfer)."""
        if name == "gain":
            return self.gain
        if is_parametric(self.transfer):
            return self.transfer.get(name)
        raise KeyError(f"this term's transfer exposes no parameter {name!r}")

    def with_value(self, name, value):
        """A copy with one knob set (own ``gain``, else rebuilt around the modified transfer)."""
        if name == "gain":
            return replace(self, gain=self._GAIN_DESC.validate(value, where="DynamicResponseTerm"))
        if is_parametric(self.transfer):
            return replace(self, transfer=self.transfer.with_value(name, value))
        raise KeyError(f"this term's transfer exposes no parameter {name!r}")


@dataclass
class DynamicSource:
    """How a source term's fluctuation responds to the unsteady flow ``S(omega)``.

    The modulated source quantity fluctuates as a sum over :class:`DynamicResponseTerm`::

        q'(omega) = q_mean * sum_k  term_k.gain * F_k(omega) * (phi'_k / phi_bar_k)

    For ``target="Qdot"`` (a flame) ``q'`` is the unsteady heat release [W], stamped
    onto the downstream edge's total-enthalpy (energy) row as ``q'/mdot``.  For
    ``target="mdot"`` (a mass source) ``q'`` is the unsteady injected mass-flow [kg/s],
    stamped onto the source element's mass row and, in proportion to the injection
    velocity ``u_inj``, its momentum row (as ``q' u_inj / A``).  ``u_inj`` is zero by
    default, so a quiescent injector perturbs mass only -- the momentum stamp appears
    only for an injector given a non-zero velocity.

    Parameters
    ----------
    terms : list of DynamicResponseTerm
        The transfer-function terms summed to form the response.
    target : {"Qdot", "mdot"}, optional
        Which source quantity is modulated (default ``"Qdot"``).
    q_mean : float, optional
        Mean of the modulated quantity used to de-normalize the fractional response:
        ``Q_bar`` [W] for ``target="Qdot"`` or the mean injected ``mdot`` [kg/s] for
        ``target="mdot"``.  ``None`` (default) auto-derives it from the converged mean
        flame/source (the flame's mean enthalpy rise times ``mdot`` for heat release,
        the element's injected ``mdot`` for a mass source); pass a value to override.
    """

    terms: List[DynamicResponseTerm] = field(default_factory=list)
    target: str = "Qdot"
    q_mean: Optional[float] = None

    def __post_init__(self):
        if self.target not in _TARGETS:
            raise ValueError(f"target must be one of {_TARGETS}; got {self.target!r}")
        self.terms = [t if isinstance(t, DynamicResponseTerm) else DynamicResponseTerm(**t) for t in self.terms]
        if not self.terms:
            raise ValueError("a DynamicSource needs at least one DynamicResponseTerm")
        if self.q_mean is not None:
            self.q_mean = float(self.q_mean)

    @property
    def analytic(self) -> bool:
        """Whether every term is analytically continuable (usable for stability)."""
        return all(t.transfer.analytic for t in self.terms)

    @property
    def max_delay(self) -> float:
        """Longest pure time delay across the terms [s] (for the stability contour clamp)."""
        return max((t.transfer.max_delay for t in self.terms), default=0.0)

    # -- scalar-parameter protocol (see nefes.elements.parametric) -----------------------
    # A single-term source promotes its term's knobs to the top level ("gain", "tau"); a
    # multi-term source prefixes them with the term index ("terms[0].gain").

    @staticmethod
    def _split_term(name):
        m = _TERM_ADDRESS.match(name)
        return (int(m.group(1)), m.group(2)) if m else (None, name)

    def param_descriptors(self):
        """The terms' knobs: promoted names for a single term, ``terms[k].`` prefixes otherwise."""
        if len(self.terms) == 1:
            return self.terms[0].param_descriptors()
        descs = []
        for k, t in enumerate(self.terms):
            for d in t.param_descriptors():
                descs.append(replace(d, name=f"terms[{k}].{d.name}"))
        return tuple(descs)

    def get(self, name):
        """Current value of one knob (promoted or ``terms[k].``-prefixed)."""
        k, leaf = self._split_term(name)
        if k is None:
            if len(self.terms) != 1:
                raise KeyError(f"this source has {len(self.terms)} terms; address the knob as 'terms[k].{name}'")
            k = 0
        if not 0 <= k < len(self.terms):
            raise KeyError(f"term index {k} out of range; this source has {len(self.terms)} terms")
        return self.terms[k].get(leaf)

    def with_value(self, name, value):
        """A copy with one knob set on the addressed term."""
        k, leaf = self._split_term(name)
        if k is None:
            if len(self.terms) != 1:
                raise KeyError(f"this source has {len(self.terms)} terms; address the knob as 'terms[k].{name}'")
            k = 0
        if not 0 <= k < len(self.terms):
            raise KeyError(f"term index {k} out of range; this source has {len(self.terms)} terms")
        terms = list(self.terms)
        terms[k] = terms[k].with_value(leaf, value)
        return replace(self, terms=terms)


# -- convenience constructors ----------------------------------------------


def heat_release_response(transfer, ref_edge, *, quantity="u", gain=1.0, q_mean=None) -> DynamicSource:
    """A single-term heat-release response (the common velocity-FTF flame).

    Equivalent to ``DynamicSource([DynamicResponseTerm(transfer, ref_edge, quantity,
    gain)], target="Qdot", q_mean=q_mean)``.

    Parameters
    ----------
    transfer : TransferFunction or (n, tau) or number or callable
        The frequency response ``F``; coerced via :func:`as_transfer`.
    ref_edge : int
        Edge whose fluctuation drives the response (e.g. the edge just upstream of the flame).
    quantity : str, optional
        Reference quantity: one of ``"u"``, ``"p"``, ``"rho"``, ``"mdot"`` or a composition
        scalar ``"Z:<name>"`` (default ``"u"``).
    gain : float, optional
        Real scalar multiplier on the term (default ``1.0``).
    q_mean : float, optional
        Mean heat release [W]; ``None`` (default) auto-derives it from the mean flame.

    Returns
    -------
    DynamicSource
    """
    return DynamicSource(terms=[DynamicResponseTerm(transfer, ref_edge, quantity, gain)], target="Qdot", q_mean=q_mean)


def n_tau_flame(n, tau, ref_edge, *, quantity="u", q_mean=None) -> DynamicSource:
    """The headline ``n-tau`` flame: heat release ``= q_mean * n e^{-i omega tau} * (phi'_ref/phi_bar)``.

    Parameters
    ----------
    n : float or complex
        Interaction index; the gain of the model is ``abs(n)``.
    tau : float
        Time lag [s].
    ref_edge : int
        Reference edge (typically the edge just upstream of the flame).
    quantity : str, optional
        Reference quantity (default ``"u"``).
    q_mean : float, optional
        Mean heat release [W]; ``None`` auto-derives it from the mean flame.
    """
    return heat_release_response(NTau(n, tau), ref_edge, quantity=quantity, q_mean=q_mean)


def mass_flow_response(transfer, ref_edge, *, quantity="u", gain=1.0, mdot_mean=None) -> DynamicSource:
    """A single-term injected-mass-flow response (e.g. a velocity-modulated fuel feed).

    Equivalent to ``DynamicSource([...], target="mdot", q_mean=mdot_mean)``.

    Parameters
    ----------
    transfer : TransferFunction or (n, tau) or number or callable
        The frequency response ``F``; coerced via :func:`as_transfer`.
    ref_edge : int
        Edge whose fluctuation drives the injected-mass modulation.
    quantity : str, optional
        Reference quantity (default ``"u"``); see :func:`heat_release_response`.
    gain : float, optional
        Real scalar multiplier on the term (default ``1.0``).
    mdot_mean : float, optional
        Mean injected mass flow [kg/s]; ``None`` (default) auto-derives it from the element.

    Returns
    -------
    DynamicSource
    """
    return DynamicSource(
        terms=[DynamicResponseTerm(transfer, ref_edge, quantity, gain)], target="mdot", q_mean=mdot_mean
    )
