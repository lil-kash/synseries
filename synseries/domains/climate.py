"""
synseries.domains.climate
Domain-valid Climate, Weather, and Meteorological time-series generator:
- Solar Irradiance with astronomical zenith angle
- Ambient temperature with thermal inertia phase-lag behind solar peak
- Relative Humidity obeying psychrometric inverse temperature coupling
- Barometric pressure with synoptic pressure fronts and atmospheric tides
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from synseries.core.signal_generators import colored_noise, ornstein_uhlenbeck


def generate_climate_weather(
    duration_days: float = 14.0,
    sampling_freq: str = "1h",
    season: str = "summer",  # 'summer', 'winter', 'monsoon'
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate authentic multi-sensor meteorological data:
    - solar_radiation_w_m2
    - air_temperature_celsius
    - relative_humidity_pct
    - barometric_pressure_hpa
    - wind_speed_m_s
    """
    rng = np.random.default_rng(seed)
    n_steps = int(duration_days * 24)
    timestamps = pd.date_range("2026-07-01 00:00:00", periods=n_steps, freq=sampling_freq)
    t_hours = np.linspace(0, duration_days * 24.0, n_steps)

    base_temp = 28.0 if season == "summer" else (12.0 if season == "winter" else 24.0)
    temp_amplitude = 7.0 if season == "summer" else 5.0
    hour = t_hours % 24.0

    # 1. Solar Radiation (W/m^2): peaks at 12:00 solar noon
    solar_noon_dist = np.maximum(0.0, np.cos(np.pi * (hour - 12.0) / 12.0))
    solar_noon_dist[hour < 6.0] = 0.0
    solar_noon_dist[hour > 18.0] = 0.0
    max_solar = 950.0 if season == "summer" else 550.0
    solar = max_solar * (solar_noon_dist**1.2)

    # 2. Air Temperature: peaks at ~15:00 (3-hour thermal inertia lag behind solar noon)
    diurnal_temp = temp_amplitude * np.cos(2 * np.pi * (hour - 15.0) / 24.0)
    # Synoptic multi-day cold/warm front wave (period ~ 5 days)
    front_wave = 3.5 * np.sin(2 * np.pi * t_hours / (5 * 24.0))
    temp = base_temp + diurnal_temp + front_wave + colored_noise(n_steps, kind="pink", std=0.6, seed=seed)

    # 3. Relative Humidity (%):
    # Clausius-Clapeyron relation: Warm air holds more vapor, so RH is lowest when Temp is highest
    humidity_base = 65.0 if season != "monsoon" else 88.0
    humidity = humidity_base - 2.2 * diurnal_temp + colored_noise(n_steps, kind="pink", std=2.5, seed=seed + 1 if seed else None)
    if season == "monsoon":
        humidity += 15.0

    # 4. Barometric Pressure (hPa):
    # Synoptic pressure wave (low pressure system indicates storm) + 12h atmospheric tide
    baro_tide = 1.2 * np.sin(2 * np.pi * hour / 12.0)
    synoptic_pressure = -6.0 * np.sin(2 * np.pi * t_hours / (5 * 24.0))  # Inverse to warm front
    pressure = 1013.25 + synoptic_pressure + baro_tide + colored_noise(n_steps, kind="white", std=0.3, seed=seed + 2 if seed else None)

    # 5. Wind Speed (m/s): Weibull distributed
    u = rng.uniform(0.001, 0.999, n_steps)
    wind = 4.5 * (-np.log(u)) ** (1.0 / 1.8) + (hour >= 12) * (hour <= 18) * 1.5

    # Bounds clipping
    solar = np.clip(solar, 0.0, 1200.0)
    temp = np.clip(temp, -30.0, 50.0)
    humidity = np.clip(humidity, 10.0, 100.0)
    pressure = np.clip(pressure, 940.0, 1060.0)
    wind = np.clip(wind, 0.0, 45.0)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "solar_radiation_w_m2": np.round(solar, 1),
            "air_temperature_celsius": np.round(temp, 2),
            "relative_humidity_pct": np.round(humidity, 1),
            "barometric_pressure_hpa": np.round(pressure, 2),
            "wind_speed_m_s": np.round(wind, 2),
        }
    )
    df = df.set_index("timestamp")
    return df
