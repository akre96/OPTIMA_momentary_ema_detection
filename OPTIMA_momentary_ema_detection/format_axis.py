""" Group of functions that make plots prettier

"""

import seaborn as sns
from matplotlib import rcParams


def setup_style():
    rcParams["font.family"] = "Arial"
    rcParams["figure.dpi"] = 300


def despine_thicken_axes(
    ax,
    lw: float = 2,
    fontsize: float = 20,
    x_rotation: float = 0,
    y_rotation: float = 0,
    x_tick_fontsize: float | None = None,
    y_tick_fontsize: float | None = None,
    xlabel_fontsize: float | None = None,
    ylabel_fontsize: float | None = None,
    heatmap: bool = False,
    grid: bool = False,
):
    """Despine axes, rotate x or y, thicken axes

    Arguments:
        ax -- matplotlib axis to modify

    Keyword Arguments:
        lw {float} -- line width for axes (default: {4})
        fontsize {float} --  fontsize for axes labels/ticks (default: {30})
        x_rotation {float} -- rotation in degrees for x-axis ticks (default: {0})
        y_rotation {float} -- rotation in degrees for y-axis ticks (default: {0})

    Returns:
        ax -- modified input axis
    """
    # Change axis tick thickness
    ax.xaxis.set_tick_params(width=lw, length=lw * 2, color="black")
    ax.yaxis.set_tick_params(width=lw, length=lw * 2, color="black")
    ax.spines["top"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")

    # Make axis lines thicker
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(lw)

    # Set fontsize variables
    if x_tick_fontsize is None:
        x_tick_fontsize = fontsize
    if y_tick_fontsize is None:
        y_tick_fontsize = fontsize
    if xlabel_fontsize is None:
        xlabel_fontsize = fontsize
    if ylabel_fontsize is None:
        ylabel_fontsize = fontsize

    # Make labels correct fontsize
    ax.set_ylabel(ax.get_ylabel(), fontsize=xlabel_fontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=ylabel_fontsize)

    # Remove spines (top/right axes) if not heatmap
    if not heatmap:
        sns.despine()

    # Rotate tick labels and set thickness
    for var in ["x", "y"]:
        fs = y_tick_fontsize
        rot = y_rotation
        if var == "x":
            fs = x_tick_fontsize
            rot = x_rotation
        ax.tick_params(axis=var, which="major", labelsize=fs)
        ax.tick_params(axis=var, which="minor", labelsize=fs * 0.8)
        ax.tick_params(axis=var, rotation=rot)
    if grid:
        ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)
    else:
        ax.grid(False)
    return ax
