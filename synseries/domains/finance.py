"""
synseries.domains.finance
Domain-valid quantitative finance and market time-series generator:
- Geometric Brownian Motion with Merton Jump Diffusion and stochastic volatility
- Multi-asset correlated portfolios (Equities, Forex, Crypto, Bonds)
- OHLCV Candlestick synthesis respecting financial no-arbitrage invariants
- Volume-Weighted Average Price (VWAP) and Bid-Ask spreads
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from synseries.core.signal_generators import merton_jump_diffusion, colored_noise
from synseries.core.correlations import induce_correlation_matrix


def generate_ohlcv(
    symbol: str = "AAPL",
    n_bars: int = 500,
    frequency: str = "1h",
    initial_price: float = 150.0,
    annual_volatility: float = 0.25,
    annual_drift: float = 0.10,
    jump_intensity: float = 2.0,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate authentic OHLCV candlestick time series with stochastic intraday excursions.
    Ensures absolute mathematical invariants:
      High >= max(Open, Close)
      Low <= min(Open, Close)
      Volume > 0 and correlated with absolute price volatility
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01 09:30:00", periods=n_bars, freq=frequency)

    # Time step in years (assuming 252 trading days, 6.5 hours/day)
    dt = 1.0 / (252.0 * 6.5) if frequency == "1h" else 1.0 / 252.0

    # Simulate sub-bar price paths to realistically sample high, low, open, close
    sub_steps_per_bar = 10
    total_sub_steps = n_bars * sub_steps_per_bar
    sub_dt = dt / sub_steps_per_bar

    # Jump diffusion for realistic tail events
    sub_prices = merton_jump_diffusion(
        n_steps=total_sub_steps,
        dt=sub_dt,
        s0=initial_price,
        mu=annual_drift,
        sigma=annual_volatility,
        jump_lambda=jump_intensity,
        jump_mu=-0.015,
        jump_sigma=0.04,
        seed=seed,
    )

    reshaped = sub_prices.reshape((n_bars, sub_steps_per_bar))

    opens = reshaped[:, 0]
    closes = reshaped[:, -1]
    highs = np.max(reshaped, axis=1)
    lows = np.min(reshaped, axis=1)

    # Ensure wicks exist even if price is flat
    min_wick = 0.001 * opens
    highs = np.maximum(highs, np.maximum(opens, closes) + rng.exponential(min_wick))
    lows = np.minimum(lows, np.minimum(opens, closes) - rng.exponential(min_wick))
    lows = np.maximum(lows, 0.01)  # No negative prices

    # Realistic volume modeling:
    # Volume is elevated when price changes are large (high volatility = high volume)
    returns = np.abs((closes - opens) / opens)
    base_volume = rng.lognormal(mean=12.0, sigma=0.4, size=n_bars)
    volatility_volume_boost = 1.0 + 15.0 * returns
    volumes = (base_volume * volatility_volume_boost).astype(int)

    # Calculate VWAP (Volume Weighted Average Price)
    typical_price = (highs + lows + closes) / 3.0
    cum_vol = np.cumsum(volumes)
    cum_pv = np.cumsum(typical_price * volumes)
    vwap = np.round(cum_pv / np.maximum(1, cum_vol), 2)

    # Bid-Ask spread (tight for high volume, wide for low volume)
    spread_pct = 0.0005 + 0.002 * (1.0 / (1.0 + returns * 50.0))
    bid = np.round(closes * (1.0 - spread_pct / 2.0), 2)
    ask = np.round(closes * (1.0 + spread_pct / 2.0), 2)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(closes, 2),
            "volume": volumes,
            "vwap": vwap,
            "bid": bid,
            "ask": ask,
        }
    )
    df = df.set_index("timestamp")
    return df


def generate_portfolio(
    symbols: List[str] = ["SPY", "QQQ", "GLD", "TLT", "BTC"],
    n_days: int = 365,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate multi-asset correlated price series (Equities, Gold, Treasuries, Crypto).
    Realistic correlation matrix:
    - SPY and QQQ have high positive correlation (~0.90)
    - SPY and TLT (Treasuries) have negative correlation (~ -0.35)
    - SPY and GLD (Gold) have low correlation (~0.05)
    - BTC has moderate correlation with high volatility
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=n_days, freq="B")  # Business days
    d = len(symbols)

    # Target correlation matrix for [SPY, QQQ, GLD, TLT, BTC]
    target_corr = np.array([
        [ 1.00,  0.88,  0.08, -0.35,  0.40],
        [ 0.88,  1.00,  0.05, -0.40,  0.48],
        [ 0.08,  0.05,  1.00,  0.25,  0.10],
        [-0.35, -0.40,  0.25,  1.00, -0.20],
        [ 0.40,  0.48,  0.10, -0.20,  1.00],
    ])

    # Crop or adjust matrix to match provided symbols count
    target_corr = target_corr[:d, :d]

    # Generate independent standard Gaussian returns
    daily_returns_raw = rng.normal(0.0, 1.0, size=(n_days, d))

    # Apply Cholesky correlation
    corr_returns = induce_correlation_matrix(daily_returns_raw, target_corr)

    # Asset specific annual volatilities and drifts
    annual_vols = [0.16, 0.22, 0.14, 0.12, 0.65][:d]
    annual_drifts = [0.10, 0.14, 0.06, 0.03, 0.35][:d]
    initial_prices = [450.0, 380.0, 190.0, 95.0, 42000.0][:d]

    dt = 1.0 / 252.0
    prices = np.zeros((n_days, d))

    for j in range(d):
        sigma = annual_vols[j]
        mu = annual_drifts[j]
        scaled_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * corr_returns[:, j]
        prices[:, j] = initial_prices[j] * np.exp(np.cumsum(scaled_ret))

    df_dict = {"timestamp": timestamps}
    for j, sym in enumerate(symbols[:d]):
        df_dict[f"{sym}_price"] = np.round(prices[:, j], 2)

    df = pd.DataFrame(df_dict)
    df = df.set_index("timestamp")
    return df
