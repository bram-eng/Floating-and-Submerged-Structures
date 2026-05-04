"""
tests/test_eva.py
=================
Unit tests for extreme_value_analysis.py and metocean_analysis.py.

Run from the Chapter1_MetOcean directory:

    python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Make sure the parent directory is on the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from extreme_value_analysis import (
    fit_gev,
    gev_return_value,
    fit_gpd,
    gpd_return_value,
    mean_residual_life,
    select_threshold_mrl,
    block_maxima_eva,
    pot_eva,
    EVAResult,
    RETURN_PERIODS,
)
from metocean_analysis import (
    statistical_summary,
    annual_maxima,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """30-year synthetic 3-hourly DataFrame with Hs and U10."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1993-01-01", periods=30 * 365 * 8, freq="3h")
    n = len(dates)
    hs = np.abs(1.0 + 0.7 * rng.standard_normal(n))
    u10 = np.abs(7.0 + 3.0 * rng.standard_normal(n))
    hs = np.clip(hs, 0.05, None)
    u10 = np.clip(u10, 0.1, None)
    df = pd.DataFrame({"Hs": hs, "U10": u10}, index=dates)
    return df


@pytest.fixture()
def annual_max_sample() -> np.ndarray:
    """30 years of simulated annual maxima drawn from GEV(c=0.1, loc=3, scale=0.7)."""
    rng = np.random.default_rng(7)
    from scipy.stats import genextreme
    return genextreme.rvs(0.1, loc=3.0, scale=0.7, size=30, random_state=rng)


# ---------------------------------------------------------------------------
# GEV fitting
# ---------------------------------------------------------------------------

class TestGEVFit:
    def test_fit_returns_three_params(self, annual_max_sample):
        c, loc, scale = fit_gev(annual_max_sample)
        assert isinstance(c, float)
        assert isinstance(loc, float)
        assert isinstance(scale, float)

    def test_scale_positive(self, annual_max_sample):
        _, _, scale = fit_gev(annual_max_sample)
        assert scale > 0

    def test_return_value_monotone(self, annual_max_sample):
        c, loc, scale = fit_gev(annual_max_sample)
        rv = gev_return_value(RETURN_PERIODS, c, loc, scale)
        # Longer return period → larger quantile
        assert np.all(np.diff(rv) >= 0)

    def test_return_value_reasonable(self, annual_max_sample):
        c, loc, scale = fit_gev(annual_max_sample)
        rv_100 = float(gev_return_value(100, c, loc, scale))
        # 100-year value should exceed the sample mean
        assert rv_100 > annual_max_sample.mean()


# ---------------------------------------------------------------------------
# GPD fitting
# ---------------------------------------------------------------------------

class TestGPDFit:
    def test_fit_returns_two_params(self):
        rng = np.random.default_rng(1)
        excesses = rng.exponential(scale=1.5, size=200)
        xi, sigma = fit_gpd(excesses)
        assert isinstance(xi, float)
        assert sigma > 0

    def test_return_value_monotone(self):
        rng = np.random.default_rng(2)
        excesses = rng.exponential(scale=1.5, size=300)
        xi, sigma = fit_gpd(excesses)
        threshold = 0.5
        rv = gpd_return_value(
            RETURN_PERIODS, threshold, xi, sigma,
            n_years=30, n_exceedances=300,
        )
        assert np.all(np.diff(rv) >= 0), "Return values should be non-decreasing"


# ---------------------------------------------------------------------------
# MRL / threshold selection
# ---------------------------------------------------------------------------

class TestMRL:
    def test_mrl_lengths_match(self):
        rng = np.random.default_rng(3)
        data = rng.exponential(2.0, size=500)
        thresholds, mrl, mrl_se = mean_residual_life(data)
        assert len(thresholds) == len(mrl) == len(mrl_se)

    def test_threshold_within_data_range(self):
        rng = np.random.default_rng(4)
        data = rng.exponential(2.0, size=500)
        u = select_threshold_mrl(data)
        assert data.min() <= u <= data.max()


# ---------------------------------------------------------------------------
# Block Maxima EVA (integration)
# ---------------------------------------------------------------------------

class TestBlockMaximaEVA:
    def test_returns_eva_result(self, synthetic_df):
        result = block_maxima_eva(synthetic_df, "Hs")
        assert result is not None
        assert isinstance(result, EVAResult)
        assert result.method == "BM-GEV"

    def test_ci_brackets_estimate(self, synthetic_df):
        result = block_maxima_eva(synthetic_df, "Hs", n_bootstrap=200)
        assert result is not None
        # Ignore NaN entries (can arise at very short return periods in bootstrap)
        valid = ~np.isnan(result.ci_lower) & ~np.isnan(result.ci_upper)
        assert np.all(result.ci_lower[valid] <= result.return_values[valid] + 1e-6)
        assert np.all(result.ci_upper[valid] >= result.return_values[valid] - 1e-6)

    def test_missing_variable_returns_none(self, synthetic_df):
        result = block_maxima_eva(synthetic_df, "Vc")  # not in fixture
        assert result is None

    def test_return_values_length(self, synthetic_df):
        result = block_maxima_eva(synthetic_df, "Hs")
        assert result is not None
        assert len(result.return_values) == len(RETURN_PERIODS)


# ---------------------------------------------------------------------------
# POT EVA (integration)
# ---------------------------------------------------------------------------

class TestPOTEVA:
    def test_returns_eva_result(self, synthetic_df):
        result = pot_eva(synthetic_df, "Hs")
        assert result is not None
        assert isinstance(result, EVAResult)
        assert result.method == "POT-GPD"

    def test_custom_threshold(self, synthetic_df):
        result = pot_eva(synthetic_df, "Hs", threshold=2.5)
        assert result is not None

    def test_missing_variable_returns_none(self, synthetic_df):
        result = pot_eva(synthetic_df, "Tp")  # not in fixture
        assert result is None

    def test_100yr_exceeds_10yr(self, synthetic_df):
        result = pot_eva(synthetic_df, "Hs")
        assert result is not None
        idx_10 = np.where(result.return_periods == 10)[0][0]
        idx_100 = np.where(result.return_periods == 100)[0][0]
        assert result.return_values[idx_100] >= result.return_values[idx_10]


# ---------------------------------------------------------------------------
# Statistical summary (metocean_analysis)
# ---------------------------------------------------------------------------

class TestStatisticalSummary:
    def test_returns_dataframe(self, synthetic_df):
        summary = statistical_summary(synthetic_df)
        assert isinstance(summary, pd.DataFrame)
        assert not summary.empty

    def test_contains_mean_and_std(self, synthetic_df):
        summary = statistical_summary(synthetic_df)
        assert "mean" in summary.index
        assert "std" in summary.index

    def test_empty_input(self):
        empty_df = pd.DataFrame(columns=["Hs"])
        summary = statistical_summary(empty_df)
        assert summary.empty or summary.shape[1] == 0


class TestAnnualMaxima:
    def test_one_max_per_year(self, synthetic_df):
        am = annual_maxima(synthetic_df)
        assert len(am) == 30  # 30 years in the fixture

    def test_max_not_less_than_mean(self, synthetic_df):
        am = annual_maxima(synthetic_df)
        # Each annual maximum must be ≥ the mean of that specific year
        for year, row in am.iterrows():
            year_data = synthetic_df.loc[str(year.year), "Hs"]
            assert row["Hs"] >= year_data.mean()
