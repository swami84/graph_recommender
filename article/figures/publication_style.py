"""Shared visual system for publication figures."""

from matplotlib import font_manager
import matplotlib.pyplot as plt


INK = "#263238"
MUTED = "#667085"
GRID = "#D9DEE5"
NAVY = "#3977A8"
TEAL = "#2A9D8F"
CORAL = "#E76F51"
GOLD = "#E9B44C"
PURPLE = "#756BB1"
LIGHT = "#EEF2F6"
RATING_COLORS = ["#C65D57", "#D98A62", "#D8B65C", "#75A879", "#2A9D8F"]


def apply_style() -> None:
    # DejaVu Sans is preferred because it includes the star glyph used in rating axes.
    for candidate in ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans", "Liberation Sans"]:
        if any(font.name == candidate for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "font.size": 10,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": GRID,
        "grid.alpha": 0.55,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    })


def clean_axis(axis, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)


def title(fig, headline: str, subtitle: str) -> None:
    fig.suptitle(headline, fontsize=17, fontweight="bold", color=INK, y=1.03)
    fig.text(0.5, 0.975, subtitle, ha="center", va="top", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.93])


def save(fig, path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=240)
    plt.close(fig)
