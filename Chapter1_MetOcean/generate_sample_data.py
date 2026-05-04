"""
generate_sample_data.py
=======================
Generate a synthetic 30-year hourly MetOcean record for the Gulf of Finland
(Helsinki–Tallinn corridor, approx. 59.8 °N, 25.0 °E).

The synthetic data is calibrated to broadly match published Baltic Sea
metocean statistics and is intended **for testing and demonstration only**.
Do NOT use it for actual design.

Run from the Chapter1_MetOcean directory:

    python generate_sample_data.py

Output: ``data/sample_gulf_of_finland.csv``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).parent / "data" / "sample_gulf_of_finland.csv"
SEED = 2024
START = "1993-01-01"
END = "2022-12-31 23:00"
FREQ = "3h"  # 3-hourly sea-state records


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dates = pd.date_range(START, END, freq=FREQ)
    n = len(dates)

    # --- Day-of-year for seasonality ---
    doy = dates.day_of_year.values  # 1–366
    season = np.cos(2 * np.pi * (doy - 355) / 365.25)  # peaks in Jan (winter)

    # --- Significant Wave Height (Hs) ---
    # Weibull-like: seasonal mean ~0.5 m (summer) to ~1.5 m (winter)
    hs_mean = 1.0 + 0.5 * season  # 0.5–1.5 m
    hs_std = 0.6 + 0.3 * season   # 0.3–0.9 m
    # Use Rayleigh-like generation: Hs ~ Weibull(k=1.8, lambda=hs_mean)
    scale = hs_mean / (np.sqrt(np.pi / 2))  # approximate for k=2
    hs = rng.weibull(1.8, size=n) * (hs_mean / (np.exp(np.log(1.8) / 1.8 - np.log(1.8))))
    hs = np.abs(hs_mean + hs_std * rng.standard_normal(n))
    hs = np.clip(hs, 0.05, None)
    # Add rare extreme events (~10/year)
    n_extremes = int(n * 10 / (365.25 * 8))
    extreme_idx = rng.choice(n, size=n_extremes, replace=False)
    hs[extreme_idx] = rng.uniform(3.5, 6.5, size=n_extremes)

    # --- Peak Period (Tp) – correlated with Hs ---
    # Tp ≈ 5.2 * sqrt(Hs) with scatter
    tp = 5.2 * np.sqrt(hs) + rng.normal(0, 0.8, size=n)
    tp = np.clip(tp, 2.0, 14.0)

    # --- Wind Speed at 10 m (U10) ---
    u10_mean = 7.0 + 2.5 * season  # 4.5–9.5 m/s
    u10 = np.abs(u10_mean + (3.0 + 1.0 * season) * rng.standard_normal(n))
    u10[extreme_idx] = rng.uniform(18.0, 28.0, size=n_extremes)
    u10 = np.clip(u10, 0.1, None)

    # --- Current Speed (Vc) ---
    vc = rng.rayleigh(0.15, size=n)
    vc = np.clip(vc, 0.01, 1.2)

    df = pd.DataFrame({
        "datetime": dates,
        "Hs": np.round(hs, 3),
        "Tp": np.round(tp, 2),
        "U10": np.round(u10, 2),
        "Vc": np.round(vc, 3),
    })
    return df


def main() -> None:
    print("Generating synthetic MetOcean data …")
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Written {len(df):,} records to {OUTPUT_PATH}")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
