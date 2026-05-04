"""
metocean_analysis.py
====================
Load MetOcean CSV data, perform basic quality control, and produce a
statistical summary for each variable.

Expected CSV format (one file per variable, or a combined file):

    datetime,Hs,Tp,U10,Vc
    2000-01-01 00:00,1.23,8.4,7.2,0.15
    ...

Columns
-------
datetime : ISO-8601 timestamp
Hs       : Significant wave height  [m]
Tp       : Peak wave period         [s]
U10      : Wind speed at 10 m       [m/s]
Vc       : Surface current speed    [m/s]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Variables to analyse and their display labels / units
VARIABLES = {
    "Hs": ("Significant Wave Height", "m"),
    "Tp": ("Peak Wave Period", "s"),
    "U10": ("Wind Speed at 10 m", "m/s"),
    "Vc": ("Surface Current Speed", "m/s"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metocean_data(filepath: str | Path) -> pd.DataFrame:
    """Load MetOcean data from a CSV file.

    Parameters
    ----------
    filepath:
        Path to the CSV file.  The file must contain a 'datetime' column
        (parseable by :func:`pandas.to_datetime`) and at least one of the
        variable columns defined in ``VARIABLES``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by datetime with float columns for each variable.
    """
    df = pd.read_csv(filepath, parse_dates=["datetime"], index_col="datetime")
    df = df.sort_index()

    # Keep only recognised variable columns
    cols_present = [c for c in VARIABLES if c in df.columns]
    if not cols_present:
        raise ValueError(
            f"No recognised variable columns found in {filepath}.\n"
            f"Expected one or more of: {list(VARIABLES.keys())}"
        )
    df = df[cols_present].astype(float)

    # Basic QC: drop rows where all values are NaN
    df = df.dropna(how="all")
    return df


def load_all_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and concatenate all CSV files in *data_dir*.

    Returns an empty DataFrame with the expected columns if no CSV files are
    found (so that the rest of the script can still run without crashing).
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(
            f"[WARNING] No CSV files found in {data_dir}.\n"
            "          Place your MetOcean CSV files there and re-run."
        )
        return pd.DataFrame(columns=list(VARIABLES.keys()))

    frames = [load_metocean_data(f) for f in csv_files]
    df = pd.concat(frames).sort_index()
    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep="first")]
    print(f"Loaded {len(df):,} records from {len(csv_files)} file(s).")
    return df


# ---------------------------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------------------------

def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of key statistics for each variable in *df*."""
    if df.empty:
        return pd.DataFrame()

    stats = df.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return stats


def annual_maxima(df: pd.DataFrame) -> pd.DataFrame:
    """Return the annual maximum of each variable."""
    if df.empty:
        return pd.DataFrame()
    return df.resample("YE").max()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_time_series(df: pd.DataFrame, save: bool = True) -> None:
    """Plot time-series of all available variables."""
    if df.empty:
        return

    cols = [c for c in VARIABLES if c in df.columns]
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 3 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        label, unit = VARIABLES[col]
        ax.plot(df.index, df[col], linewidth=0.5, color="steelblue")
        ax.set_ylabel(f"{label}\n[{unit}]", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Date")
    fig.suptitle("MetOcean Time Series – Helsinki–Tallinn", fontsize=12, y=1.01)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "time_series.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


def plot_histograms(df: pd.DataFrame, save: bool = True) -> None:
    """Plot histograms / probability density estimates for each variable."""
    if df.empty:
        return

    cols = [c for c in VARIABLES if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        label, unit = VARIABLES[col]
        data = df[col].dropna()
        ax.hist(data, bins=60, density=True, color="steelblue", alpha=0.7,
                edgecolor="white", linewidth=0.3)
        ax.set_xlabel(f"{label} [{unit}]", fontsize=9)
        ax.set_ylabel("Probability Density", fontsize=9)
        ax.set_title(col)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("MetOcean Variable Distributions", fontsize=12)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "histograms.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


def plot_annual_maxima(df: pd.DataFrame, save: bool = True) -> None:
    """Bar-chart of annual maxima for each variable."""
    if df.empty:
        return

    am = annual_maxima(df)
    cols = [c for c in VARIABLES if c in am.columns]

    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        label, unit = VARIABLES[col]
        ax.bar(am.index.year, am[col], color="steelblue", edgecolor="white",
               linewidth=0.5)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel(f"Annual Max {label} [{unit}]", fontsize=9)
        ax.set_title(col)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Annual Maxima", fontsize=12)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "annual_maxima.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_all_data()

    print("\n=== Statistical Summary ===")
    summary = statistical_summary(df)
    if not summary.empty:
        print(summary.to_string())
    else:
        print("No data available for summary.")

    print("\n=== Annual Maxima ===")
    am = annual_maxima(df)
    if not am.empty:
        print(am.to_string())

    # Plots
    plot_time_series(df)
    plot_histograms(df)
    plot_annual_maxima(df)

    print("\nDone. Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
