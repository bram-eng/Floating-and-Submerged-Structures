"""
extreme_value_analysis.py
=========================
Extreme Value Analysis (EVA) of MetOcean variables.

Two complementary methods are implemented:

1. **Block Maxima (BM)** – fits a Generalized Extreme Value (GEV)
   distribution to annual (or seasonal) block maxima.
2. **Peaks-Over-Threshold (POT)** – fits a Generalized Pareto Distribution
   (GPD) to threshold exceedances; the threshold is chosen with a Mean
   Residual Life (MRL) plot and stability plots.

Return-period quantiles are estimated for T = 1, 5, 10, 25, 50, 100 years.

Usage
-----
    python extreme_value_analysis.py

The script reads data from ``Chapter1_MetOcean/data/`` via
:mod:`metocean_analysis`.  All figures are written to
``Chapter1_MetOcean/figures/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Local import
from metocean_analysis import (
    DATA_DIR,
    FIGURES_DIR,
    VARIABLES,
    load_all_data,
    annual_maxima,
)

RETURN_PERIODS = np.array([1, 2, 5, 10, 25, 50, 100])  # years


# ---------------------------------------------------------------------------
# Helper data class
# ---------------------------------------------------------------------------

class EVAResult(NamedTuple):
    """Container for EVA output for one variable and one method."""
    variable: str
    method: str          # 'BM-GEV' or 'POT-GPD'
    params: dict         # fitted distribution parameters
    return_periods: np.ndarray
    return_values: np.ndarray
    ci_lower: np.ndarray  # 95 % confidence interval lower bound
    ci_upper: np.ndarray  # 95 % confidence interval upper bound


# ---------------------------------------------------------------------------
# Block Maxima / GEV
# ---------------------------------------------------------------------------

def fit_gev(annual_max: np.ndarray) -> tuple[float, float, float]:
    """Fit a GEV distribution to annual maxima.

    Returns
    -------
    c, loc, scale
        GEV shape (c), location, and scale parameters.
        Note: :func:`scipy.stats.genextreme` uses the sign convention
        ``c = –ξ``, so positive *c* corresponds to a Weibull tail (ξ < 0).
    """
    c, loc, scale = stats.genextreme.fit(annual_max)
    return c, loc, scale


def gev_return_value(
    T: float | np.ndarray,
    c: float,
    loc: float,
    scale: float,
) -> np.ndarray:
    """Return the GEV quantile for return period *T* (years)."""
    p_exceed = 1.0 / np.asarray(T, dtype=float)
    return stats.genextreme.ppf(1 - p_exceed, c, loc=loc, scale=scale)


def block_maxima_eva(
    df: pd.DataFrame,
    variable: str,
    n_bootstrap: int = 500,
) -> EVAResult | None:
    """Perform Block Maxima EVA for *variable* using annual maxima.

    Parameters
    ----------
    df:
        Full MetOcean DataFrame (datetime-indexed).
    variable:
        Column name to analyse.
    n_bootstrap:
        Number of bootstrap resamples for confidence intervals.

    Returns
    -------
    EVAResult or None if insufficient data.
    """
    if variable not in df.columns:
        return None

    am = df[variable].resample("YE").max().dropna().values
    if len(am) < 5:
        print(f"[WARNING] Fewer than 5 annual maxima for {variable}; skipping BM-GEV.")
        return None

    c, loc, scale = fit_gev(am)
    rv = gev_return_value(RETURN_PERIODS, c, loc, scale)

    # Bootstrap confidence intervals
    boot_rv = np.empty((n_bootstrap, len(RETURN_PERIODS)))
    rng = np.random.default_rng(42)
    for i in range(n_bootstrap):
        sample = rng.choice(am, size=len(am), replace=True)
        bc, bloc, bscale = fit_gev(sample)
        boot_rv[i] = gev_return_value(RETURN_PERIODS, bc, bloc, bscale)

    ci_lower = np.percentile(boot_rv, 2.5, axis=0)
    ci_upper = np.percentile(boot_rv, 97.5, axis=0)

    return EVAResult(
        variable=variable,
        method="BM-GEV",
        params={"c (−ξ)": c, "loc (μ)": loc, "scale (σ)": scale},
        return_periods=RETURN_PERIODS,
        return_values=rv,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


# ---------------------------------------------------------------------------
# Peaks-Over-Threshold / GPD
# ---------------------------------------------------------------------------

def mean_residual_life(
    data: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Mean Residual Life (MRL) for a range of thresholds.

    Returns
    -------
    thresholds, mrl, mrl_se
    """
    data = np.sort(data)
    if thresholds is None:
        thresholds = np.linspace(data.min(), np.percentile(data, 95), 100)

    mrl = np.empty(len(thresholds))
    mrl_se = np.empty(len(thresholds))
    for i, u in enumerate(thresholds):
        excess = data[data > u] - u
        if len(excess) < 2:
            mrl[i] = np.nan
            mrl_se[i] = np.nan
        else:
            mrl[i] = excess.mean()
            mrl_se[i] = excess.std(ddof=1) / np.sqrt(len(excess))

    return thresholds, mrl, mrl_se


def select_threshold_mrl(data: np.ndarray) -> float:
    """Heuristic threshold selection: first point of approximate linearity
    in the MRL plot (90th percentile as a robust default).
    """
    return float(np.percentile(data, 90))


def fit_gpd(excesses: np.ndarray) -> tuple[float, float]:
    """Fit a GPD to threshold excesses.

    Returns
    -------
    shape (ξ), scale (σ)
    """
    xi, _, sigma = stats.genpareto.fit(excesses, floc=0)
    return xi, sigma


def gpd_return_value(
    T: float | np.ndarray,
    threshold: float,
    xi: float,
    sigma: float,
    n_years: float,
    n_exceedances: int,
) -> np.ndarray:
    """Return GPD quantile for return period *T* years.

    Parameters
    ----------
    n_years:
        Total record length in years.
    n_exceedances:
        Number of threshold exceedances.
    """
    T = np.asarray(T, dtype=float)
    lambda_u = n_exceedances / n_years  # mean exceedances per year
    # Probability of non-exceedance in one year
    p_exceed = 1.0 / T
    # Conditional exceedance probability given threshold exceedance
    p_cond = p_exceed / lambda_u
    p_cond = np.clip(p_cond, 0, 1)
    # GPD quantile
    excess_q = stats.genpareto.ppf(1 - p_cond, xi, scale=sigma)
    return threshold + excess_q


def pot_eva(
    df: pd.DataFrame,
    variable: str,
    threshold: float | None = None,
    n_bootstrap: int = 500,
) -> EVAResult | None:
    """Perform POT / GPD EVA for *variable*.

    Parameters
    ----------
    threshold:
        Exceedance threshold.  If *None*, chosen automatically as the 90th
        percentile of the data (see :func:`select_threshold_mrl`).
    """
    if variable not in df.columns:
        return None

    data = df[variable].dropna().values
    if threshold is None:
        threshold = select_threshold_mrl(data)

    excesses = data[data > threshold] - threshold
    n_exceedances = len(excesses)
    if n_exceedances < 10:
        print(f"[WARNING] Fewer than 10 exceedances for {variable} at u={threshold:.3f}; "
              "skipping POT-GPD.")
        return None

    record_years = (df.index[-1] - df.index[0]).days / 365.25
    xi, sigma = fit_gpd(excesses)
    rv = gpd_return_value(RETURN_PERIODS, threshold, xi, sigma,
                          record_years, n_exceedances)

    # Bootstrap confidence intervals
    boot_rv = np.empty((n_bootstrap, len(RETURN_PERIODS)))
    rng = np.random.default_rng(42)
    for i in range(n_bootstrap):
        sample = rng.choice(excesses, size=len(excesses), replace=True)
        bxi, bsigma = fit_gpd(sample)
        boot_rv[i] = gpd_return_value(RETURN_PERIODS, threshold, bxi, bsigma,
                                      record_years, n_exceedances)

    ci_lower = np.percentile(boot_rv, 2.5, axis=0)
    ci_upper = np.percentile(boot_rv, 97.5, axis=0)

    return EVAResult(
        variable=variable,
        method="POT-GPD",
        params={
            "threshold (u)": threshold,
            "shape (ξ)": xi,
            "scale (σ)": sigma,
            "n_exceedances": n_exceedances,
            "record_years": record_years,
        },
        return_periods=RETURN_PERIODS,
        return_values=rv,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mrl(data: np.ndarray, variable: str, save: bool = True) -> None:
    """Mean Residual Life plot for threshold selection."""
    thresholds, mrl, mrl_se = mean_residual_life(data)
    threshold_chosen = select_threshold_mrl(data)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, mrl, color="steelblue", linewidth=1.5, label="MRL")
    ax.fill_between(
        thresholds,
        mrl - 1.96 * mrl_se,
        mrl + 1.96 * mrl_se,
        alpha=0.25,
        color="steelblue",
        label="95 % CI",
    )
    ax.axvline(threshold_chosen, color="tomato", linestyle="--",
               label=f"Selected threshold = {threshold_chosen:.2f}")
    label, unit = VARIABLES.get(variable, (variable, ""))
    ax.set_xlabel(f"Threshold [{unit}]")
    ax.set_ylabel("Mean Residual Life")
    ax.set_title(f"MRL Plot – {label}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"mrl_{variable}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


def plot_return_period(result: EVAResult, save: bool = True) -> None:
    """Return period plot with 95 % bootstrap confidence band."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogx(result.return_periods, result.return_values,
                "o-", color="steelblue", linewidth=2, markersize=5,
                label=result.method)
    ax.fill_between(
        result.return_periods,
        result.ci_lower,
        result.ci_upper,
        alpha=0.25,
        color="steelblue",
        label="95 % CI",
    )

    label, unit = VARIABLES.get(result.variable, (result.variable, ""))
    ax.set_xlabel("Return Period [years]")
    ax.set_ylabel(f"{label} [{unit}]")
    ax.set_title(f"Return Period Curve – {label} ({result.method})")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xlim(result.return_periods[0], result.return_periods[-1])
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"return_period_{result.variable}_{result.method.replace('-','_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


def plot_probability_plot(
    data: np.ndarray,
    variable: str,
    method: str,
    save: bool = True,
) -> None:
    """Probability plot (Q-Q plot against the fitted distribution)."""
    if method == "BM-GEV":
        am = pd.Series(data).resample("YE").max().dropna().values \
            if isinstance(data, pd.Series) else data
        c, loc, scale = fit_gev(am)
        dist = stats.genextreme(c, loc=loc, scale=scale)
        values = np.sort(am)
    else:
        threshold = select_threshold_mrl(data)
        excesses = data[data > threshold] - threshold
        xi, sigma = fit_gpd(excesses)
        dist = stats.genpareto(xi, scale=sigma)
        values = np.sort(excesses)

    n = len(values)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = dist.ppf(probs)

    label, unit = VARIABLES.get(variable, (variable, ""))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical, values, s=20, alpha=0.6, color="steelblue")
    lims = [
        min(theoretical.min(), values.min()),
        max(theoretical.max(), values.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="1:1 line")
    ax.set_xlabel(f"Theoretical Quantiles [{unit}]")
    ax.set_ylabel(f"Sample Quantiles [{unit}]")
    ax.set_title(f"Probability Plot – {label} ({method})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"prob_plot_{variable}_{method.replace('-','_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_eva_table(results: list[EVAResult]) -> None:
    """Print a formatted table of return-period values."""
    header = f"{'Variable':<8}  {'Method':<10}  " + \
             "  ".join(f"T={T:>4}yr" for T in RETURN_PERIODS)
    print("\n=== Return-Period Estimates ===")
    print(header)
    print("-" * len(header))
    for r in results:
        if r is None:
            continue
        _, unit = VARIABLES.get(r.variable, (r.variable, "?"))
        row_vals = "  ".join(f"{v:>8.3f}" for v in r.return_values)
        print(f"{r.variable:<8}  {r.method:<10}  {row_vals}  [{unit}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_all_data()

    results: list[EVAResult | None] = []

    for var in VARIABLES:
        if var not in df.columns:
            continue

        print(f"\n--- {var} ---")
        data = df[var].dropna().values

        # MRL plot (threshold selection aid)
        plot_mrl(data, var)

        # Block Maxima EVA
        bm = block_maxima_eva(df, var)
        if bm:
            print(f"  BM-GEV params: {bm.params}")
            plot_return_period(bm)
            plot_probability_plot(
                df[var].resample("YE").max().dropna().values,
                var,
                "BM-GEV",
            )
            results.append(bm)

        # POT EVA
        pot = pot_eva(df, var)
        if pot:
            print(f"  POT-GPD params: {pot.params}")
            plot_return_period(pot)
            results.append(pot)

    print_eva_table([r for r in results if r is not None])
    print("\nDone. Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
