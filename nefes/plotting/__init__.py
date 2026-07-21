"""Plotly presentation layer for Nefes.

A single home for everything Plotly-related: a light and a dark theme matching the
documentation and the Nemo interface, so every figure across the examples and notebooks
shares one consistent look.  The routines here draw with the Nefes theme already; call
:func:`set_theme` to switch modes, which also makes the theme Plotly's default so
hand-built figures match::

    from nefes.plotting import set_theme, palette
    set_theme("dark")        # every subsequent figure, shipped or hand-built, turns dark
    palette().accent         # the ember accent of the active mode

It also hosts the complex-matrix viewers used to read transfer / scattering
matrices in a notebook (magnitude over phase, with presets for the 2x2 acoustic
and 3x3 full perturbation networks)::

    from nefes.plotting import plot_transfer_matrix
    plot_transfer_matrix(resp.transfer_matrix(0, 1), resp.freqs).show()

Labels are MathJax (``$...$``) by default.  Where MathJax does not render (a plain
kernel, a static export), call :func:`use_latex(False) <nefes.plotting.use_latex>` to
switch every Nefes figure to a Unicode plain-text fallback.

This subpackage renders with Plotly, which the core solver does not require and which
ships in the ``viz`` extra (``pip install nefes[viz]``).  Importing ``nefes.plotting``
succeeds without Plotly; the missing-dependency error is deferred until a figure is built.
"""

from .complex_matrix import (
    plot_complex_matrix,
    plot_scattering_matrix,
    plot_transfer_matrix,
    scattering_axis_labels,
)
from .continuation import plot_fit, plot_pole_map
from .labels import detex, latex_enabled, mathify, tex, tex_text, use_latex
from .modeshape import AnimSeries, animate_mode_shape
from .sensitivity import plot_sensitivities
from .spectrum import plot_mode_shape, plot_spectrum
from .theme import (
    COLORWAY,
    DARK,
    FONT_FAMILY,
    LIGHT,
    NEFES_TEMPLATE_NAME,
    Palette,
    colorway,
    nefes_template,
    palette,
    set_theme,
    theme_mode,
    use_nefes_theme,
)
from .topology import plot_network_topology
from .transfer_function import plot_transfer_function

__all__ = [
    "COLORWAY",
    "FONT_FAMILY",
    "NEFES_TEMPLATE_NAME",
    "Palette",
    "LIGHT",
    "DARK",
    "set_theme",
    "theme_mode",
    "palette",
    "colorway",
    "nefes_template",
    "use_nefes_theme",
    "use_latex",
    "latex_enabled",
    "mathify",
    "tex",
    "detex",
    "tex_text",
    "plot_complex_matrix",
    "plot_transfer_matrix",
    "plot_scattering_matrix",
    "scattering_axis_labels",
    "plot_transfer_function",
    "plot_fit",
    "plot_pole_map",
    "plot_sensitivities",
    "plot_spectrum",
    "plot_mode_shape",
    "animate_mode_shape",
    "AnimSeries",
    "plot_network_topology",
]
