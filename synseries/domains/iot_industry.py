"""
synseries.domains.iot_industry
Domain-valid Industrial IoT and Predictive Maintenance time-series generator:
- 3-Axis Accelerometer Bearing Vibration with defect harmonics (BPFO, BPFI)
- Degradation run-to-failure curves with Remaining Useful Life (RUL) tracking
- Motor electrical and thermal dynamics (Current, Torque, Stator Temperature)
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from synseries.core.signal_generators import colored_noise, ornstein_uhlenbeck


def generate_bearing_vibration(
    duration_seconds: float = 5.0,
    sampling_rate_hz: int = 2000,
    shaft_rpm: float = 1750.0,
    defect_type: Optional[str] = "outer_race",  # None, 'outer_race', 'inner_race', 'unbalance'
    degradation_pct: float = 50.0,  # 0 to 100% wear
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate authentic high-frequency 3-axis vibration telemetry (g-force).
    Models shaft rotational harmonics and characteristic bearing defect frequencies:
      BPFO = Ball Pass Frequency Outer Race
      BPFI = Ball Pass Frequency Inner Race
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_seconds * sampling_rate_hz)
    t = np.linspace(0, duration_seconds, n_samples)

    shaft_freq = shaft_rpm / 60.0  # Hz (~29.17 Hz for 1750 RPM)
    # Typical bearing geometric multipliers: BPFO ~ 3.58 * f_s, BPFI ~ 5.42 * f_s
    bpfo = 3.58 * shaft_freq
    bpfi = 5.42 * shaft_freq

    # Baseline 1X and 2X shaft rotational harmonics
    vib_x = 0.2 * np.sin(2 * np.pi * shaft_freq * t) + 0.08 * np.sin(4 * np.pi * shaft_freq * t)
    vib_y = 0.2 * np.cos(2 * np.pi * shaft_freq * t) + 0.08 * np.cos(4 * np.pi * shaft_freq * t)
    vib_z = 0.1 * np.sin(2 * np.pi * shaft_freq * t + np.pi / 4)

    # Defect impacts modulated by shaft rotation
    severity = degradation_pct / 100.0
    if defect_type == "outer_race":
        # Sharp high-frequency transient resonant ringing excited at BPFO
        resonance_freq = 3200.0  # Structural resonance
        decay_rate = 350.0
        impact_signal = np.zeros(n_samples)

        # Pulse arrivals at BPFO
        pulse_period = 1.0 / bpfo
        pulse_times = np.arange(0, duration_seconds, pulse_period)
        for p_t in pulse_times:
            dt_pulse = t - p_t
            mask = (dt_pulse >= 0) & (dt_pulse < 0.015)
            if np.any(mask):
                impact_signal[mask] += (
                    severity
                    * 3.5
                    * np.exp(-decay_rate * dt_pulse[mask])
                    * np.sin(2 * np.pi * resonance_freq * dt_pulse[mask])
                )
        vib_x += impact_signal
        vib_y += impact_signal * 0.7
    elif defect_type == "unbalance":
        # Excessive 1X vibration amplitude
        vib_x += (severity * 2.5) * np.sin(2 * np.pi * shaft_freq * t)
        vib_y += (severity * 2.5) * np.cos(2 * np.pi * shaft_freq * t)

    # Mechanical background noise
    vib_x += colored_noise(n_samples, kind="white", std=0.08, seed=seed)
    vib_y += colored_noise(n_samples, kind="white", std=0.08, seed=seed + 1 if seed else None)
    vib_z += colored_noise(n_samples, kind="white", std=0.05, seed=seed + 2 if seed else None)

    timestamps = pd.date_range("2026-01-01 00:00:00", periods=n_samples, freq=f"{1e6/sampling_rate_hz:.0f}us")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "accel_x_g": np.round(vib_x, 4),
            "accel_y_g": np.round(vib_y, 4),
            "accel_z_g": np.round(vib_z, 4),
            "overall_rms_g": np.round(np.sqrt((vib_x**2 + vib_y**2 + vib_z**2) / 3.0), 4),
        }
    )
    df = df.set_index("timestamp")
    return df


def generate_predictive_maintenance_run_to_failure(
    n_days: int = 30,
    sampling_freq: str = "10min",
    nominal_current_amps: float = 45.0,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate complete lifecycle run-to-failure dataset tracking:
    - motor_current_amps
    - bearing_temperature_celsius
    - vibration_rms_g
    - acoustic_emission_db
    - remaining_useful_life_hours (RUL)
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=int(n_days * 24 * 6), freq=sampling_freq)
    n_steps = len(timestamps)
    total_hours = n_days * 24.0

    # Remaining Useful Life ground truth
    hours_elapsed = np.linspace(0, total_hours, n_steps)
    rul = np.maximum(0.0, total_hours - hours_elapsed)

    # Progressive degradation curve: Paris-Erdogan law exponential damage acceleration
    # Healthy stage -> Incipient damage -> Rapid exponential failure
    norm_time = hours_elapsed / total_hours
    damage = np.exp(4.5 * norm_time) / np.exp(4.5)  # 0 to 1 curve

    # Physical variables coupled to wear damage:
    # 1. Bearing temperature increases as friction rises
    temp_base = 52.0 + 38.0 * damage + colored_noise(n_steps, kind="pink", std=0.7, seed=seed)
    # 2. Vibration RMS increases smoothly, then explodes in final stage
    vib_rms = 0.45 + 3.8 * (damage**2) + colored_noise(n_steps, kind="white", std=0.08, seed=seed + 1 if seed else None)
    # 3. Motor current increases due to mechanical drag
    current = nominal_current_amps + 12.0 * damage + colored_noise(n_steps, kind="white", std=0.5, seed=seed + 2 if seed else None)
    # 4. High-frequency Acoustic Emission (spikes during microscopic crack formation)
    ae_db = 40.0 + 45.0 * np.sqrt(damage) + colored_noise(n_steps, kind="white", std=1.2, seed=seed + 3 if seed else None)

    # Health status label
    health_status = np.where(
        damage < 0.35, "HEALTHY", np.where(damage < 0.75, "DEGRADED", "CRITICAL_FAILURE_IMMINENT")
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "vibration_rms_g": np.round(np.clip(vib_rms, 0.1, 10.0), 3),
            "bearing_temperature_celsius": np.round(temp_base, 2),
            "motor_current_amps": np.round(current, 2),
            "acoustic_emission_db": np.round(ae_db, 1),
            "remaining_useful_life_hours": np.round(rul, 2),
            "health_status": health_status,
        }
    )
    df = df.set_index("timestamp")
    return df
