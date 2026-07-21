"""Light and dark Plotly themes for Nefes figures.

The palettes mirror the documentation theme (``docs/_theme/nefes-light.scss`` and its dark
counterpart) and the Nemo interface, so a figure dropped into a page or a panel looks native
there: the same ink and gray scale, the same ember accent, the same IBM Plex type.

Importing this module registers three templates with Plotly: ``"nefes-light"``, ``"nefes-dark"``
and ``"nefes"``, the last being an alias for whichever mode is active.  Every plotting routine
in :mod:`nefes.plotting` draws with ``"nefes"``, so figures follow the active mode with no call
on the user's part.  :func:`set_theme` switches the mode (and points the Plotly default at it);
:func:`palette` returns the colours of the active mode for figures assembled by hand.

This module exports :func:`set_theme`, :func:`theme_mode`, :func:`palette`, :func:`colorway`,
:func:`nefes_template` and :class:`Palette`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ._deps import go, pio

NEFES_TEMPLATE_NAME = "nefes"
LIGHT_TEMPLATE_NAME = "nefes-light"
DARK_TEMPLATE_NAME = "nefes-dark"

ThemeMode = Literal["light", "dark"]

# Default body font for all Plotly text; the documentation ships IBM Plex, and the
# remaining families cover kernels and static exports where it is not installed.
FONT_FAMILY = '"IBM Plex Sans", Arial, Helvetica, sans-serif'

# Figure title: centered, bold, slightly below the Plotly 18 pt default.
_TITLE_FONT_SIZE = 16
_TICK_FONT_SIZE = 13
# Default Plotly standoff is 15 px; pull axis titles closer to tick labels.
_AXIS_TITLE_STANDOFF = 6
# Uniform line weight shared by every axis line, mirrored border and tick.
_LINE_W = 1.6


@dataclass(frozen=True)
class Palette:
    """Colours of one Nefes theme mode.

    Attributes
    ----------
    name : str
        Mode name, ``"light"`` or ``"dark"``.
    colorway : list of str
        Categorical series colours, in draw order.
    paper, plot, surface : str
        Figure background, plotting-area background, and the raised background used for
        legends, hover labels and inset boxes.
    ink, muted, faint : str
        Primary text, secondary text (tick labels, annotations), and the faintest labels.
    grid, axis, rule : str
        Grid hairlines, axis lines and ticks, and reference lines drawn over the data
        (zero lines, guides).
    accent : str
        Ember accent, matching the documentation link colour.
    marker_edge : str
        Outline drawn around filled markers and bars to separate them from the background.
    sequential : list
        Colour scale for continuous fields.
    """

    name: str
    colorway: list[str]
    paper: str
    plot: str
    surface: str
    ink: str
    muted: str
    faint: str
    grid: str
    axis: str
    rule: str
    accent: str
    marker_edge: str
    sequential: list = field(default_factory=list)

    @property
    def stable(self) -> str:
        """Blue, used for stable eigenvalues and for stabilizing sensitivities."""
        return self.colorway[0]

    @property
    def unstable(self) -> str:
        """Red, used for unstable eigenvalues and for destabilizing sensitivities."""
        return self.colorway[4]


LIGHT = Palette(
    name="light",
    colorway=[
        "#2563eb",  # blue
        "#ea580c",  # ember
        "#059669",  # emerald
        "#7c3aed",  # violet
        "#dc2626",  # red
        "#0284c7",  # sky
        "#ca8a04",  # amber
        "#db2777",  # pink
    ],
    paper="#ffffff",
    plot="#ffffff",
    surface="#f5f7fa",
    ink="#1f2933",
    muted="#52606d",
    faint="#7b8794",
    grid="#eceff3",
    axis="#cbd2d9",
    rule="#9aa5b1",
    accent="#ea580c",
    marker_edge="#ffffff",
    sequential=[[0.0, "#eef2ff"], [0.5, "#60a5fa"], [1.0, "#1e3a8a"]],
)

DARK = Palette(
    name="dark",
    colorway=[
        "#60a5fa",  # blue
        "#fb923c",  # ember
        "#34d399",  # emerald
        "#a78bfa",  # violet
        "#f87171",  # red
        "#38bdf8",  # sky
        "#fbbf24",  # amber
        "#f472b6",  # pink
    ],
    paper="#151a21",
    plot="#171d25",
    surface="#1b222b",
    ink="#dbe1e8",
    muted="#9aa5b1",
    faint="#7b8794",
    grid="#232c37",
    axis="#3b4653",
    rule="#5a6674",
    accent="#fb923c",
    marker_edge="#151a21",
    sequential=[[0.0, "#12233a"], [0.5, "#3b82f6"], [1.0, "#bfdbfe"]],
)

PALETTES: dict[str, Palette] = {"light": LIGHT, "dark": DARK}

# Categorical palette of the light theme, kept as a module constant so scripts and notebooks
# can colour hand-built traces consistently; palette().colorway follows the active mode.
COLORWAY = LIGHT.colorway

_mode: str = "light"


def rgba(hex_color: str, alpha: float) -> str:
    """Return ``hex_color`` as an ``rgba(...)`` string with opacity ``alpha``."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _axis(p: Palette) -> dict:
    """Shared axis styling of both modes."""
    return dict(
        showgrid=True,
        gridcolor=p.grid,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=p.axis,
        linewidth=_LINE_W,
        mirror=True,  # mirror the axis line to the opposite side -> a full box around each subplot
        ticks="outside",
        ticklen=5,
        tickwidth=_LINE_W,
        tickcolor=p.axis,
        tickfont=dict(color=p.muted, size=_TICK_FONT_SIZE),
        title=dict(font=dict(color=p.ink, size=14), standoff=_AXIS_TITLE_STANDOFF),
        automargin=True,
    )


def theme_mode() -> str:
    """Return the active theme mode, ``"light"`` or ``"dark"``.

    See Also
    --------
    set_theme : Switch the active mode.
    """
    return _mode


def palette(mode: ThemeMode | None = None) -> Palette:
    """Return the colours of a theme mode.

    Parameters
    ----------
    mode : {"light", "dark"}, optional
        Mode to look up.  Defaults to the active mode.

    Returns
    -------
    Palette
        Named colours of that mode.

    Examples
    --------
    >>> from nefes.plotting import palette
    >>> fig.add_hline(y=0.0, line_color=palette().rule)  # doctest: +SKIP
    """
    return PALETTES[_resolve(mode)]


def colorway(mode: ThemeMode | None = None) -> list[str]:
    """Return the categorical series colours of a theme mode, in draw order."""
    return list(palette(mode).colorway)


def _resolve(mode: ThemeMode | None) -> str:
    """Validate ``mode``, falling back to the active mode when it is ``None``."""
    if mode is None:
        return _mode
    if mode not in PALETTES:
        raise ValueError(f"unknown theme mode {mode!r}; expected 'light' or 'dark'")
    return mode


def nefes_template(mode: ThemeMode | None = None) -> go.layout.Template:
    """Build (without registering) the Nefes Plotly template for one mode."""
    p = palette(mode)
    return go.layout.Template(
        layout=dict(
            colorway=list(p.colorway),
            font=dict(family=FONT_FAMILY, color=p.ink, size=13),
            title=dict(
                font=dict(family=FONT_FAMILY, color=p.ink, size=_TITLE_FONT_SIZE, weight="bold"),
                x=0.5,
                xanchor="center",
            ),
            paper_bgcolor=p.paper,
            plot_bgcolor=p.plot,
            colorscale=dict(sequential=p.sequential),
            xaxis=_axis(p),
            yaxis=_axis(p),
            margin=dict(l=70, r=30, t=60, b=60),
            legend=dict(
                bgcolor=rgba(p.surface, 0.85),
                bordercolor=p.grid,
                borderwidth=1,
                font=dict(color=p.muted, size=12),
            ),
            hoverlabel=dict(
                bgcolor=p.surface,
                bordercolor=p.axis,
                font=dict(family=FONT_FAMILY, color=p.ink, size=12),
            ),
            hovermode="x unified",
        ),
        data=dict(
            scatter=[go.Scatter(line=dict(width=2.5), marker=dict(size=7, line=dict(width=0)))],
        ),
    )


def _register(default: bool = True) -> str:
    """Register both mode templates plus the ``"nefes"`` alias for the active mode."""
    pio.templates[LIGHT_TEMPLATE_NAME] = nefes_template("light")
    pio.templates[DARK_TEMPLATE_NAME] = nefes_template("dark")
    pio.templates[NEFES_TEMPLATE_NAME] = nefes_template(_mode)
    if default:
        pio.templates.default = NEFES_TEMPLATE_NAME
    return NEFES_TEMPLATE_NAME


def set_theme(mode: ThemeMode = "light") -> str:
    """Switch every Nefes figure to the light or the dark theme.

    The plotting routines of :mod:`nefes.plotting` draw with the Nefes template already, so
    they follow the mode set here without further action.  The Plotly process-wide default is
    pointed at the same template, so figures built by hand match as well.

    Parameters
    ----------
    mode : {"light", "dark"}, default "light"
        Theme mode to activate.

    Returns
    -------
    str
        Name of the active template, for callers that prefer to pass ``template=`` explicitly.

    Examples
    --------
    >>> from nefes.plotting import set_theme
    >>> set_theme("dark")  # doctest: +SKIP
    'nefes'

    See Also
    --------
    palette : Colours of the active mode, for hand-built traces.
    """
    global _mode
    _mode = _resolve(mode)
    return _register()


def use_nefes_theme(mode: ThemeMode | None = None) -> str:
    """Make the Nefes template the process-wide Plotly default.

    Equivalent to :func:`set_theme`, which is the preferred spelling; kept because it reads
    naturally at the top of a notebook and appears in the shipped examples.

    Parameters
    ----------
    mode : {"light", "dark"}, optional
        Theme mode to activate.  Defaults to leaving the active mode unchanged.

    Returns
    -------
    str
        Name of the active template.
    """
    return set_theme(_resolve(mode))


# Register on import so every shipped figure is themed without an explicit call.  The Plotly
# default is left alone here, so hand-built figures keep Plotly's look until the user asks
# otherwise.  Guarded so the module still imports without Plotly, in which case registration
# waits for the first call.
try:
    _register(default=False)
except ModuleNotFoundError:
    pass
