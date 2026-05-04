"""
plots.py
========
Shared visualisation utilities for the MetOcean & EVA analysis.

All plot functions write PNG files to the ``figures/`` sub-directory
(created automatically if it does not exist).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

_STYLE = {
    "color_main": "steelblue",
    "color_accent": "tomato",
    "alpha_fill": 0.25,
    "grid_alpha": 0.4,
    "dpi": 150,
}


# ---------------------------------------------------------------------------
# Scatter / joint plots
# ---------------------------------------------------------------------------

def plot_hs_tp_scatter(df: pd.DataFrame, save: bool = True) -> None:
    """Scatter plot of Hs vs Tp coloured by data density."""
    if "Hs" not in df.columns or "Tp" not in df.columns:
        print("[SKIP] plot_hs_tp_scatter: Hs or Tp column missing.")
        return

    hs = df["Hs"].dropna()
    tp = df["Tp"].dropna()
    common = hs.index.intersection(tp.index)
    hs, tp = hs[common].values, tp[common].values

    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(tp, hs, gridsize=60, cmap="YlOrRd", mincnt=1)
    cb = fig.colorbar(hb, ax=ax, label="Count")
    ax.set_xlabel("Peak Period $T_p$ [s]")
    ax.set_ylabel("Significant Wave Height $H_s$ [m]")
    ax.set_title("$H_s$–$T_p$ Scatter (Gulf of Finland)")
    ax.grid(True, linestyle="--", alpha=_STYLE["grid_alpha"])
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "hs_tp_scatter.png"
        fig.savefig(path, dpi=_STYLE["dpi"], bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


def plot_wind_wave_scatter(df: pd.DataFrame, save: bool = True) -> None:
    """Scatter plot of wind speed vs Hs."""
    if "Hs" not in df.columns or "U10" not in df.columns:
        print("[SKIP] plot_wind_wave_scatter: Hs or U10 column missing.")
        return

    hs = df["Hs"].dropna()
    u10 = df["U10"].dropna()
    common = hs.index.intersection(u10.index)
    hs, u10 = hs[common].values, u10[common].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(u10, hs, s=2, alpha=0.3, color=_STYLE["color_main"], rasterized=True)
    ax.set_xlabel("Wind Speed $U_{10}$ [m/s]")
    ax.set_ylabel("Significant Wave Height $H_s$ [m]")
    ax.set_title("Wind Speed vs $H_s$")
    ax.grid(True, linestyle="--", alpha=_STYLE["grid_alpha"])
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "wind_wave_scatter.png"
        fig.savefig(path, dpi=_STYLE["dpi"], bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rose / directional plots (if direction columns are present)
# ---------------------------------------------------------------------------

def plot_wave_rose(
    df: pd.DataFrame,
    direction_col: str = "Dir",
    magnitude_col: str = "Hs",
    bins: int = 16,
    save: bool = True,
) -> None:
    """Polar histogram (wave/wind rose).

    Parameters
    ----------
    direction_col:
        Name of the column containing directions in degrees (0–360, from N).
    magnitude_col:
        Column whose magnitude is plotted (e.g. Hs or U10).
    """
    if direction_col not in df.columns or magnitude_col not in df.columns:
        print(f"[SKIP] plot_wave_rose: {direction_col!r} or {magnitude_col!r} column missing.")
        return

    direction = df[direction_col].dropna()
    magnitude = df[magnitude_col].dropna()
    common = direction.index.intersection(magnitude.index)
    direction = np.deg2rad(direction[common].values)
    magnitude = magnitude[common].values

    theta_bins = np.linspace(0, 2 * np.pi, bins + 1)
    radii, _ = np.histogram(direction, bins=theta_bins, weights=magnitude)
    counts, _ = np.histogram(direction, bins=theta_bins)
    radii = np.where(counts > 0, radii / counts, 0)  # mean magnitude per sector
    theta = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    width = 2 * np.pi / bins

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.bar(theta, radii, width=width, bottom=0, alpha=0.7,
           color=_STYLE["color_main"], edgecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise, matching compass bearings
    ax.set_title(f"Mean {magnitude_col} by Direction", pad=20)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"rose_{magnitude_col}.png"
        fig.savefig(path, dpi=_STYLE["dpi"], bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Seasonal analysis
# ---------------------------------------------------------------------------

def plot_seasonal_boxplots(df: pd.DataFrame, variable: str = "Hs",
                           save: bool = True) -> None:
    """Box-plots of *variable* grouped by month."""
    if variable not in df.columns:
        print(f"[SKIP] plot_seasonal_boxplots: {variable!r} column missing.")
        return

    data = df[[variable]].dropna().copy()
    data["month"] = data.index.month
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(10, 4))
    grouped = [data.loc[data["month"] == m, variable].values for m in range(1, 13)]
    bp = ax.boxplot(grouped, patch_artist=True,
                    boxprops={"facecolor": _STYLE["color_main"], "alpha": 0.6},
                    medianprops={"color": "white", "linewidth": 2},
                    whiskerprops={"linewidth": 1},
                    capprops={"linewidth": 1},
                    flierprops={"markersize": 2, "alpha": 0.3})
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_ylabel(f"{variable}")
    ax.set_title(f"Seasonal Variation of {variable}")
    ax.grid(True, axis="y", linestyle="--", alpha=_STYLE["grid_alpha"])
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"seasonal_{variable}.png"
        fig.savefig(path, dpi=_STYLE["dpi"], bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exceedance probability
# ---------------------------------------------------------------------------

def plot_exceedance_probability(
    df: pd.DataFrame,
    variable: str = "Hs",
    save: bool = True,
) -> None:
    """Empirical exceedance probability (1 – CDF) on a log-y axis."""
    if variable not in df.columns:
        print(f"[SKIP] plot_exceedance_probability: {variable!r} column missing.")
        return

    data = np.sort(df[variable].dropna().values)
    n = len(data)
    exceedance = 1 - (np.arange(1, n + 1) - 0.5) / n

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(data, exceedance, color=_STYLE["color_main"], linewidth=1.5)
    ax.set_xlabel(f"{variable}")
    ax.set_ylabel("Exceedance Probability")
    ax.set_title(f"Empirical Exceedance Probability – {variable}")
    ax.grid(True, which="both", linestyle="--", alpha=_STYLE["grid_alpha"])
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"exceedance_{variable}.png"
        fig.savefig(path, dpi=_STYLE["dpi"], bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)
