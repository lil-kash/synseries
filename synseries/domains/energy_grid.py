"""
synseries.domains.energy_grid
Domain-valid Smart Grid and Renewable Energy time-series generator:
- Solar PV Photovoltaic generation with astronomical solar zenith and cloud occlusions
- Wind turbine power with Weibull wind speed and non-linear power curves
- Microgrid frequency stability (50Hz / 60Hz) with generator trip events
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from synseries.core.signal_generators import ornstein_uhlenbeck, colored_noise


def generate_smart_grid(
    duration_days: float = 7.0,
    sampling_freq: str = "15min",
    nominal_freq_hz: float = 50.0,
    has_grid_trip: bool = True,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate realistic multi-channel smart microgrid telemetry:
    - solar_generation_mw
    - wind_generation_mw
    - consumer_load_demand_mw
    - net_grid_power_mw
    - grid_frequency_hz (nominal 50.00 or 60.00 Hz)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(duration_days * 24 * 4)  # 15-min intervals
    timestamps = pd.date_range("2026-06-01 00:00:00", periods=n_steps, freq=sampling_freq)
    t_hours = np.linspace(0, duration_days * 24.0, n_steps)

    # 1. Solar Photovoltaic Generation (MW)
    # Peak solar capacity: 80 MW
    solar_capacity_mw = 80.0
    hour_of_day = t_hours % 24.0
    # Sun elevation angle from 06:00 to 18:00
    solar_angle = np.maximum(0.0, np.sin(np.pi * (hour_of_day - 6.0) / 12.0))
    solar_angle[hour_of_day < 6.0] = 0.0
    solar_angle[hour_of_day > 18.0] = 0.0

    # Cloud cover factor (0 = clear sky, 1 = overcast)
    cloud_cover = np.clip(
        colored_noise(n_steps, kind="pink", std=0.25, seed=seed) + 0.2,
        0.0,
        0.85,
    )
    solar_mw = solar_capacity_mw * (solar_angle**1.3) * (1.0 - cloud_cover)

    # 2. Wind Turbine Generation (MW)
    # Wind speed follows Weibull distribution (shape k=2.0 Rayleigh, scale c=8.5 m/s)
    wind_capacity_mw = 60.0
    u = rng.uniform(0.001, 0.999, n_steps)
    wind_speed = 8.5 * (-np.log(u)) ** (1.0 / 2.0)
    # Wind turbine power curve:
    # v_cut_in = 3 m/s, v_rated = 12 m/s, v_cut_out = 25 m/s
    wind_power_ratio = np.zeros(n_steps)
    mask_gen = (wind_speed >= 3.0) & (wind_speed < 12.0)
    wind_power_ratio[mask_gen] = ((wind_speed[mask_gen] - 3.0) / 9.0) ** 3
    wind_power_ratio[(wind_speed >= 12.0) & (wind_speed < 25.0)] = 1.0
    # Over 25 m/s turbines brake for safety (cut-out)
    wind_power_ratio[wind_speed >= 25.0] = 0.0
    wind_mw = wind_capacity_mw * wind_power_ratio

    # 3. Consumer Load Demand (MW)
    # Diurnal twin peak (morning 08:30 and evening 19:30)
    morning_peak = 35.0 * np.exp(-0.5 * ((hour_of_day - 8.5) / 2.0) ** 2)
    evening_peak = 45.0 * np.exp(-0.5 * ((hour_of_day - 19.5) / 2.5) ** 2)
    base_demand = 60.0 + morning_peak + evening_peak
    demand_mw = base_demand + colored_noise(n_steps, kind="pink", std=2.5, seed=seed + 1 if seed else None)

    # Net Grid Power: Generation - Demand
    net_power_mw = solar_mw + wind_mw - demand_mw

    # 4. Grid Frequency (Hz)
    # Mean-reverting around nominal with swing equation inertia
    # Negative net power (demand > gen) drags frequency down; positive lifts it
    freq = ornstein_uhlenbeck(n_steps, dt=15.0/60.0, theta=0.8, mu=nominal_freq_hz, sigma=0.012, seed=seed + 2 if seed else None)
    freq += 0.0008 * net_power_mw

    # Grid trip event (sudden generator disconnection)
    if has_grid_trip:
        trip_idx = int(n_steps * 0.6)
        if trip_idx < n_steps:
            decay_len = min(20, n_steps - trip_idx)
            freq_drop = -0.35 * np.exp(-np.linspace(0, 3, decay_len))
            freq[trip_idx : trip_idx + decay_len] += freq_drop

    # Clip frequency to plausible grid limits (49.0 - 51.0 Hz)
    freq = np.clip(freq, nominal_freq_hz - 1.2, nominal_freq_hz + 1.2)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "solar_generation_mw": np.round(solar_mw, 2),
            "wind_generation_mw": np.round(wind_mw, 2),
            "total_renewable_mw": np.round(solar_mw + wind_mw, 2),
            "demand_load_mw": np.round(demand_mw, 2),
            "net_power_balance_mw": np.round(net_power_mw, 2),
            "grid_frequency_hz": np.round(freq, 3),
        }
    )
    df = df.set_index("timestamp")
    return df
