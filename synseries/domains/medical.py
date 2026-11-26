"""
synseries.domains.medical
Domain-valid medical and clinical time-series generators:
- Continuous ECG waveform with P-Q-R-S-T morphology, HRV, and arrhythmias
- ICU Multi-Vital Signs (HR, BP Systolic/Diastolic, SpO2, Resp Rate, Temp) with physiological shock
- Continuous Glucose Monitoring (CGM) with meal digestion and insulin curves
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from synseries.core.signal_generators import generate_timestamps, colored_noise, ornstein_uhlenbeck
from synseries.core.correlations import induce_correlation_matrix
from synseries.core.anomalies import AnomalyInjector


def generate_ecg(
    duration_seconds: float = 10.0,
    sampling_rate_hz: int = 250,
    heart_rate_bpm: float = 72.0,
    hrv_level: float = 0.05,
    noise_std: float = 0.03,
    arrhythmia: Optional[str] = None,  # None, 'pvc', 'bradycardia', 'tachycardia', 'st_elevation'
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate realistic Lead II Electrocardiogram (ECG) waveform based on sum of Gaussians
    for P, Q, R, S, T waves with cardiac cycle timing and heart rate variability (HRV).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_seconds * sampling_rate_hz)
    t = np.linspace(0, duration_seconds, n_samples)
    dt = 1.0 / sampling_rate_hz

    # Cardiac waveform morphology parameters (relative angle theta_i, amplitude a_i, width b_i)
    # Canonical Lead II parameters:
    # P wave: atrial depolarization
    # Q wave: septal depolarization
    # R wave: ventricular depolarization (dominant sharp peak)
    # S wave: late ventricular depolarization
    # T wave: ventricular repolarization
    wave_params = {
        "P": {"theta": -np.pi / 3, "a": 0.25, "b": 0.25},
        "Q": {"theta": -np.pi / 12, "a": -0.20, "b": 0.10},
        "R": {"theta": 0.0, "a": 1.40, "b": 0.05},
        "S": {"theta": np.pi / 16, "a": -0.35, "b": 0.12},
        "T": {"theta": np.pi / 2, "a": 0.35, "b": 0.40},
    }

    if arrhythmia == "st_elevation":
        # Elevated ST segment (sign of acute myocardial infarction)
        wave_params["T"]["a"] = 0.65
        wave_params["S"]["a"] = 0.10
    elif arrhythmia == "bradycardia":
        heart_rate_bpm = 42.0
    elif arrhythmia == "tachycardia":
        heart_rate_bpm = 135.0

    # Build instantaneous phase with Heart Rate Variability (HRV)
    mean_rr = 60.0 / heart_rate_bpm
    # Low frequency (LF 0.1Hz) and High frequency (HF 0.25Hz respiratory sinus arrhythmia)
    hrv_signal = (
        0.7 * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 0.25 * t)
    ) * hrv_level
    instant_rr = mean_rr * (1.0 + hrv_signal)
    instant_freq = 1.0 / np.maximum(0.2, instant_rr)

    phase = np.cumsum(2 * np.pi * instant_freq * dt) % (2 * np.pi) - np.pi

    ecg_signal = np.zeros(n_samples)
    for wave, p in wave_params.items():
        d_theta = np.remainder(phase - p["theta"] + np.pi, 2 * np.pi) - np.pi
        ecg_signal += p["a"] * np.exp(-0.5 * (d_theta / p["b"]) ** 2)

    # Intersperse Premature Ventricular Contractions (PVC) if requested
    if arrhythmia == "pvc":
        n_pvcs = max(1, int(duration_seconds / 4.0))
        pvc_locs = rng.choice(np.arange(int(n_samples * 0.1), int(n_samples * 0.9)), size=n_pvcs)
        for loc in pvc_locs:
            window = int(sampling_rate_hz * 0.25)
            s_idx = max(0, loc - window)
            e_idx = min(n_samples, loc + window)
            pvc_shape = -1.8 * np.sin(np.linspace(0, np.pi, e_idx - s_idx)) ** 2
            ecg_signal[s_idx:e_idx] += pvc_shape

    # Baseline wander (respiration-induced slow drift) + high-frequency muscle tremor noise
    baseline_wander = 0.08 * np.sin(2 * np.pi * 0.25 * t)
    noise = colored_noise(n_samples, kind="pink", std=noise_std, seed=seed)

    voltage_mv = ecg_signal + baseline_wander + noise

    timestamps = pd.date_range("2026-01-01 08:00:00", periods=n_samples, freq=f"{int(1000/sampling_rate_hz)}ms")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "lead_ii_voltage_mv": np.round(voltage_mv, 4),
            "instant_heart_rate_bpm": np.round(instant_freq * 60.0, 1),
        }
    )
    df = df.set_index("timestamp")
    return df


def generate_icu_vitals(
    duration_hours: float = 12.0,
    sampling_freq: str = "5s",
    condition: str = "normal",  # 'normal', 'septic_shock', 'hypoxia', 'hypertensive_crisis'
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate multivariate ICU telemetry:
    - heart_rate (bpm)
    - blood_pressure_systolic (mmHg)
    - blood_pressure_diastolic (mmHg)
    - spo2 (oxygen saturation %)
    - respiration_rate (breaths/min)
    - body_temperature (Celsius)
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=int(duration_hours * 3600 / 5), freq=sampling_freq)
    n_steps = len(timestamps)
    dt = 5.0 / 60.0  # minutes

    # Baseline healthy physiological parameters (Mean-reverting Ornstein-Uhlenbeck)
    hr = ornstein_uhlenbeck(n_steps, dt=dt, theta=0.08, mu=74.0, sigma=2.5, seed=seed)
    bp_sys = ornstein_uhlenbeck(n_steps, dt=dt, theta=0.06, mu=118.0, sigma=3.0, seed=seed + 1 if seed else None)
    bp_dia = ornstein_uhlenbeck(n_steps, dt=dt, theta=0.06, mu=76.0, sigma=2.0, seed=seed + 2 if seed else None)
    spo2 = 98.0 - np.abs(colored_noise(n_steps, kind="white", std=0.5, seed=seed + 3 if seed else None))
    resp = ornstein_uhlenbeck(n_steps, dt=dt, theta=0.1, mu=15.0, sigma=0.8, seed=seed + 4 if seed else None)
    temp = ornstein_uhlenbeck(n_steps, dt=dt, theta=0.02, mu=36.8, sigma=0.04, seed=seed + 5 if seed else None)

    # Induce physiologically valid correlations:
    # High BP correlates positively with Diastolic BP (~0.85)
    # Systolic and Heart rate baroreflex
    data_stack = np.column_stack([hr, bp_sys, bp_dia, resp])
    corr_target = np.array([
        [1.0,  0.25, 0.20, 0.35],
        [0.25, 1.0,  0.88, 0.15],
        [0.20, 0.88, 1.0,  0.10],
        [0.35, 0.15, 0.10, 1.0 ],
    ])
    correlated = induce_correlation_matrix(data_stack, corr_target)
    hr, bp_sys, bp_dia, resp = correlated[:, 0], correlated[:, 1], correlated[:, 2], correlated[:, 3]

    # Pathology dynamics
    if condition == "septic_shock":
        # Sepsis develops halfway through: temp spikes, HR tachycardic, BP collapses (vasodilation)
        sepsis_onset = int(n_steps * 0.4)
        ramp = np.zeros(n_steps)
        ramp[sepsis_onset:] = np.linspace(0, 1, n_steps - sepsis_onset)
        temp += ramp * 2.4       # reaches ~39.2 C
        hr += ramp * 45.0        # reaches ~125 bpm
        bp_sys -= ramp * 38.0    # collapses to ~80 mmHg
        bp_dia -= ramp * 28.0    # collapses to ~48 mmHg
        resp += ramp * 12.0      # tachypnea ~28 breaths/min
        spo2 -= ramp * 5.0       # mild hypoxia ~92%
    elif condition == "hypoxia":
        # Acute respiratory failure
        onset = int(n_steps * 0.5)
        duration = int(n_steps * 0.2)
        end = min(n_steps, onset + duration)
        dip = np.sin(np.linspace(0, np.pi, end - onset))
        spo2[onset:end] -= dip * 15.0  # drops to 83%
        hr[onset:end] += dip * 35.0    # compensatory tachycardia
        resp[onset:end] += dip * 14.0  # severe air hunger
    elif condition == "hypertensive_crisis":
        crisis_start = int(n_steps * 0.3)
        bp_sys[crisis_start:] += 65.0   # > 180 mmHg
        bp_dia[crisis_start:] += 35.0   # > 110 mmHg

    # Physical clipping bounds to ensure physiological validity
    hr = np.clip(hr, 30.0, 220.0)
    bp_sys = np.clip(bp_sys, 40.0, 240.0)
    bp_dia = np.clip(bp_dia, 25.0, bp_sys - 15.0)  # Pulse pressure guarantee
    spo2 = np.clip(spo2, 60.0, 100.0)
    resp = np.clip(resp, 4.0, 50.0)
    temp = np.clip(temp, 33.0, 42.5)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "heart_rate_bpm": np.round(hr, 1),
            "blood_pressure_systolic_mmhg": np.round(bp_sys, 1),
            "blood_pressure_diastolic_mmhg": np.round(bp_dia, 1),
            "spo2_percent": np.round(spo2, 1),
            "respiration_rate_bpm": np.round(resp, 1),
            "body_temperature_celsius": np.round(temp, 2),
        }
    )
    df = df.set_index("timestamp")
    return df


def generate_cgm(
    duration_days: float = 3.0,
    patient_type: str = "type_1_diabetic",  # 'healthy', 'type_1_diabetic', 'type_2_diabetic'
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate Continuous Glucose Monitoring (CGM) readings (e.g. Dexcom/Freestyle Libre).
    Simulates meal carbohydrate absorption peaks and insulin actions.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(duration_days * 24 * 12)  # 5-minute sampling intervals
    timestamps = pd.date_range("2026-01-01 06:00:00", periods=n_steps, freq="5min")
    t_hours = np.linspace(0, duration_days * 24, n_steps)

    # Baseline fasting glucose
    base_glucose = 90.0 if patient_type == "healthy" else (145.0 if patient_type == "type_2_diabetic" else 130.0)
    glucose = ornstein_uhlenbeck(n_steps, dt=5.0/60.0, theta=0.1, mu=base_glucose, sigma=2.0, seed=seed)

    # Dawn phenomenon (cortisol morning spike between 4 AM and 7 AM)
    hour_of_day = (t_hours + 6.0) % 24.0
    dawn_effect = 25.0 * np.exp(-0.5 * ((hour_of_day - 5.5) / 1.5) ** 2)
    if patient_type != "healthy":
        glucose += dawn_effect

    # Meals schedule: Breakfast (08:00), Lunch (13:00), Dinner (19:30)
    for day in range(int(np.ceil(duration_days))):
        meal_times = [day * 24 + 8.0, day * 24 + 13.0, day * 24 + 19.5]
        for m_time in meal_times:
            if m_time < duration_days * 24:
                carbs = rng.uniform(40, 85)
                # Absorption curve: gamma distribution shape
                delta_t = t_hours - m_time
                valid_mask = delta_t >= 0
                dt_valid = delta_t[valid_mask]
                # Excursion peak at ~60-90 mins, duration ~3 hrs
                peak_amp = carbs * (0.8 if patient_type == "healthy" else 1.9)
                curve = peak_amp * (dt_valid / 1.2) * np.exp(-dt_valid / 1.2)
                glucose[valid_mask] += curve

    # Noise
    glucose += colored_noise(n_steps, kind="pink", std=2.5, seed=seed)

    # Realistic physiological range bounds
    glucose = np.clip(glucose, 40.0, 450.0)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "glucose_mg_dl": np.round(glucose, 1),
            "glycemic_zone": np.where(
                glucose < 70, "Hypoglycemic", np.where(glucose > 180, "Hyperglycemic", "In-Target-Range")
            ),
        }
    )
    df = df.set_index("timestamp")
    return df
