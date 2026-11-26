"""
synseries.core.signal_generators
Mathematical primitives and stochastic process generators for synthetic time series.
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd


def generate_timestamps(
    start: Union[str, pd.Timestamp] = "2026-01-01 00:00:00",
    n_steps: Optional[int] = None,
    freq: str = "1s",
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DatetimeIndex:
    """Generate pandas DatetimeIndex given start and either n_steps or end."""
    if n_steps is not None:
        return pd.date_range(start=start, periods=n_steps, freq=freq)
    elif end is not None:
        return pd.date_range(start=start, end=end, freq=freq)
    else:
        raise ValueError("Either n_steps or end must be provided.")


def trend(
    t: np.ndarray,
    kind: str = "linear",
    slope: float = 0.01,
    intercept: float = 0.0,
    growth_rate: float = 0.001,
    carrying_capacity: float = 100.0,
) -> np.ndarray:
    """
    Generate deterministic trend component.
    t: normalized or integer time steps array [0, 1, 2, ... N-1]
    """
    t = np.asarray(t, dtype=float)
    if kind == "linear":
        return intercept + slope * t
    elif kind == "polynomial":
        return intercept + slope * t + 0.0001 * (t**2)
    elif kind == "exponential":
        return intercept * np.exp(np.clip(growth_rate * t, -50, 50))
    elif kind == "logistic":
        # S-curve saturation
        midpoint = len(t) / 2.0
        return carrying_capacity / (1.0 + np.exp(-growth_rate * (t - midpoint)))
    elif kind == "none":
        return np.full_like(t, intercept)
    else:
        raise ValueError(f"Unknown trend kind: {kind}")


def seasonality(
    t: np.ndarray,
    periods: List[float],
    amplitudes: List[float],
    phases: Optional[List[float]] = None,
    waveforms: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Generate harmonic or multi-frequency seasonal components.
    periods: in units of time steps (e.g., 24 for hourly day-cycle, 168 for weekly)
    """
    t = np.asarray(t, dtype=float)
    total_seasonal = np.zeros_like(t)

    if phases is None:
        phases = [0.0] * len(periods)
    if waveforms is None:
        waveforms = ["sine"] * len(periods)

    for period, amp, phase, wave in zip(periods, amplitudes, phases, waveforms):
        if period <= 0:
            continue
        rad = 2.0 * np.pi * t / period + phase
        if wave == "sine":
            signal = amp * np.sin(rad)
        elif wave == "cosine":
            signal = amp * np.cos(rad)
        elif wave == "sawtooth":
            signal = amp * 2.0 * (t / period - np.floor(0.5 + t / period))
        elif wave == "square":
            signal = amp * np.sign(np.sin(rad))
        elif wave == "diurnal":
            # Asymmetrical diurnal curve (e.g. daily solar heating: rapid morning rise, slow evening fall)
            mod_t = (t % period) / period
            signal = amp * np.maximum(0.0, np.sin(np.pi * mod_t) ** 1.5)
        else:
            signal = amp * np.sin(rad)
        total_seasonal += signal

    return total_seasonal


def colored_noise(
    n_samples: int,
    kind: str = "white",
    std: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate colored noise:
    - white (flat spectrum, 1/f^0)
    - pink (1/f, standard in biological rhythms, electronics)
    - brown / red (1/f^2, random walk noise)
    """
    rng = np.random.default_rng(seed)
    if kind == "white":
        return rng.normal(0.0, std, n_samples)

    # Spectral synthesis of 1/f^alpha noise
    alpha = 1.0 if kind == "pink" else (2.0 if kind in ["brown", "red"] else 0.0)
    if alpha == 0.0:
        return rng.normal(0.0, std, n_samples)

    # FFT synthesis
    frequencies = np.fft.rfftfreq(n_samples)
    frequencies[0] = 1e-6  # avoid division by zero
    s_filter = 1.0 / (frequencies ** (alpha / 2.0))
    s_filter[0] = 0.0

    # Gaussian random phases and magnitudes in frequency domain
    real_part = rng.normal(0.0, 1.0, len(frequencies)) * s_filter
    imag_part = rng.normal(0.0, 1.0, len(frequencies)) * s_filter
    fft_coeffs = real_part + 1j * imag_part

    noise = np.fft.irfft(fft_coeffs, n=n_samples)
    curr_std = np.std(noise)
    if curr_std > 1e-9:
        noise = noise * (std / curr_std)
    return noise


def ornstein_uhlenbeck(
    n_steps: int,
    dt: float = 1.0,
    theta: float = 0.15,
    mu: float = 37.0,
    sigma: float = 0.2,
    x0: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Ornstein-Uhlenbeck mean-reverting process:
    dx_t = theta * (mu - x_t) * dt + sigma * dW_t
    Essential for core biological vitals (body temp, baseline glucose)
    and physical stabilization loops (grid frequency).
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps)
    x[0] = mu if x0 is None else x0

    variance_dt = sigma * np.sqrt(dt)
    for i in range(1, n_steps):
        dx = theta * (mu - x[i - 1]) * dt + rng.normal(0.0, variance_dt)
        x[i] = x[i - 1] + dx

    return x


def merton_jump_diffusion(
    n_steps: int,
    dt: float = 1.0 / 252.0,
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.20,
    jump_lambda: float = 1.5,
    jump_mu: float = -0.02,
    jump_sigma: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Merton Jump-Diffusion model:
    dS/S = (mu - lambda * k) dt + sigma * dW_t + J * dN_t
    Produces realistic fat-tailed returns and crash jumps for financial series.
    """
    rng = np.random.default_rng(seed)
    prices = np.zeros(n_steps)
    prices[0] = s0

    # Compensator k = E[exp(J) - 1]
    k = np.exp(jump_mu + 0.5 * (jump_sigma**2)) - 1.0

    # Brownian motion increments
    dw = rng.normal(0.0, np.sqrt(dt), n_steps)

    # Poisson jump count per interval
    n_jumps = rng.poisson(jump_lambda * dt, n_steps)

    log_s = np.log(s0)
    drift = (mu - 0.5 * (sigma**2) - jump_lambda * k) * dt

    for t in range(1, n_steps):
        jump_contribution = 0.0
        if n_jumps[t] > 0:
            jump_mags = rng.normal(jump_mu, jump_sigma, n_jumps[t])
            jump_contribution = np.sum(jump_mags)

        log_s += drift + sigma * dw[t] + jump_contribution
        prices[t] = np.exp(log_s)

    return prices


def autoregressive_process(
    n_steps: int,
    ar_coeffs: List[float],
    intercept: float = 0.0,
    noise_std: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate AR(p) autoregressive process:
    X_t = c + sum(phi_i * X_{t-i}) + eps_t
    """
    rng = np.random.default_rng(seed)
    p = len(ar_coeffs)
    burn_in = max(50, p * 5)
    total_steps = n_steps + burn_in

    series = np.zeros(total_steps)
    noise = rng.normal(0.0, noise_std, total_steps)

    for i in range(p, total_steps):
        series[i] = intercept + noise[i]
        for lag in range(p):
            series[i] += ar_coeffs[lag] * series[i - 1 - lag]

    return series[burn_in:]
