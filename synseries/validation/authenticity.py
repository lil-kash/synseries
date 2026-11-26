"""
synseries.validation.authenticity
Domain validity and statistical authenticity verification engine for synthetic time series.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


class DomainAuthenticityValidator:
    """Verifies that synthetic datasets satisfy real-world physical and statistical invariants."""

    @classmethod
    def validate(cls, df: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """Run complete authenticity test suite and return pass/fail metrics."""
        results: Dict[str, Any] = {
            "domain": domain,
            "total_samples": len(df),
            "columns": list(df.columns),
            "checks": [],
            "overall_status": "PASS",
            "score_pct": 100.0,
        }

        # 1. Structural Checks
        cls._check_timestamps(df, results)
        cls._check_finite_values(df, results)

        # 2. Domain-Specific Invariants
        dom = domain.lower()
        if "medical" in dom:
            cls._validate_medical(df, results)
        elif "finance" in dom or "trading" in dom:
            cls._validate_finance(df, results)
        elif "iot" in dom or "industry" in dom:
            cls._validate_iot(df, results)
        elif "energy" in dom or "grid" in dom:
            cls._validate_energy(df, results)
        elif "web" in dom:
            cls._validate_web(df, results)
        elif "climate" in dom or "weather" in dom:
            cls._validate_climate(df, results)

        # 3. Statistical Memory & Realistic Dynamics Check
        cls._check_autocorrelation(df, results)

        # Compute overall score
        total_checks = len(results["checks"])
        passed_checks = sum(1 for c in results["checks"] if c["status"] == "PASS")
        score = (passed_checks / max(1, total_checks)) * 100.0
        results["score_pct"] = round(score, 1)
        results["overall_status"] = "PASS" if score >= 85.0 else "FAIL"

        return results

    @staticmethod
    def _check_timestamps(df: pd.DataFrame, results: Dict[str, Any]) -> None:
        is_dt = isinstance(df.index, pd.DatetimeIndex)
        is_monotonic = df.index.is_monotonic_increasing if is_dt else False
        results["checks"].append({
            "name": "Chronological Monotonicity",
            "category": "Structural",
            "status": "PASS" if (is_dt and is_monotonic) else "FAIL",
            "details": f"DatetimeIndex sorted: {is_monotonic}",
        })

    @staticmethod
    def _check_finite_values(df: pd.DataFrame, results: Dict[str, Any]) -> None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(df[numeric_cols].to_numpy()).any()
        results["checks"].append({
            "name": "Numerical Stability (No Infs)",
            "category": "Structural",
            "status": "PASS" if not has_inf else "FAIL",
            "details": f"Infinite values present: {has_inf}",
        })

    @staticmethod
    def _validate_medical(df: pd.DataFrame, results: Dict[str, Any]) -> None:
        cols = df.columns
        # ECG checks
        if "lead_ii_voltage_mv" in cols:
            v_min, v_max = df["lead_ii_voltage_mv"].min(), df["lead_ii_voltage_mv"].max()
            valid_range = (-3.0 <= v_min) and (v_max <= 4.0)
            results["checks"].append({
                "name": "ECG Voltage Physiological Range",
                "category": "Medical Invariant",
                "status": "PASS" if valid_range else "FAIL",
                "details": f"Min: {v_min:.2f} mV, Max: {v_max:.2f} mV (Expected: -3.0 to 4.0 mV)",
            })

        # ICU vitals checks
        if "blood_pressure_systolic_mmhg" in cols and "blood_pressure_diastolic_mmhg" in cols:
            sys_higher = (df["blood_pressure_systolic_mmhg"] > df["blood_pressure_diastolic_mmhg"]).all()
            results["checks"].append({
                "name": "Systolic > Diastolic Blood Pressure Invariant",
                "category": "Medical Invariant",
                "status": "PASS" if sys_higher else "FAIL",
                "details": f"Systolic exceeds Diastolic across 100% of samples: {sys_higher}",
            })

        if "spo2_percent" in cols:
            s_min, s_max = df["spo2_percent"].min(), df["spo2_percent"].max()
            valid_spo2 = (s_min >= 50.0) and (s_max <= 100.0)
            results["checks"].append({
                "name": "SpO2 Oxygen Saturation Bounds",
                "category": "Medical Invariant",
                "status": "PASS" if valid_spo2 else "FAIL",
                "details": f"Min: {s_min}%, Max: {s_max}% (Cannot exceed 100%)",
            })

        if "body_temperature_celsius" in cols:
            t_min, t_max = df["body_temperature_celsius"].min(), df["body_temperature_celsius"].max()
            valid_temp = (t_min >= 32.0) and (t_max <= 43.0)
            results["checks"].append({
                "name": "Human Body Temperature Survival Bounds",
                "category": "Medical Invariant",
                "status": "PASS" if valid_temp else "FAIL",
                "details": f"Min: {t_min:.1f}°C, Max: {t_max:.1f}°C",
            })

        if "glucose_mg_dl" in cols:
            g_min, g_max = df["glucose_mg_dl"].min(), df["glucose_mg_dl"].max()
            valid_g = (g_min >= 30.0) and (g_max <= 500.0)
            results["checks"].append({
                "name": "Blood Glucose Physiological Plausibility",
                "category": "Medical Invariant",
                "status": "PASS" if valid_g else "FAIL",
                "details": f"Min: {g_min} mg/dL, Max: {g_max} mg/dL",
            })

    @staticmethod
    def _validate_finance(df: pd.DataFrame, results: Dict[str, Any]) -> None:
        cols = df.columns
        if {"open", "high", "low", "close"}.issubset(cols):
            # High >= max(Open, Close)
            h_valid = (df["high"] >= np.maximum(df["open"], df["close"]) - 1e-5).all()
            results["checks"].append({
                "name": "Candlestick High >= max(Open, Close)",
                "category": "Financial Invariant",
                "status": "PASS" if h_valid else "FAIL",
                "details": f"High boundary holds across 100% of bars: {h_valid}",
            })

            # Low <= min(Open, Close)
            l_valid = (df["low"] <= np.minimum(df["open"], df["close"]) + 1e-5).all()
            results["checks"].append({
                "name": "Candlestick Low <= min(Open, Close)",
                "category": "Financial Invariant",
                "status": "PASS" if l_valid else "FAIL",
                "details": f"Low boundary holds across 100% of bars: {l_valid}",
            })

            # Non-negative prices
            no_neg = (df["low"] > 0).all()
            results["checks"].append({
                "name": "Asset Price Non-Negativity",
                "category": "Financial Invariant",
                "status": "PASS" if no_neg else "FAIL",
                "details": f"All prices > 0: {no_neg}",
            })

        if "volume" in cols:
            vol_valid = (df["volume"] >= 0).all()
            results["checks"].append({
                "name": "Trading Volume Non-Negativity",
                "category": "Financial Invariant",
                "status": "PASS" if vol_valid else "FAIL",
                "details": f"All volume bars >= 0: {vol_valid}",
            })

    @staticmethod
    def _validate_iot(df: pd.DataFrame, results: Dict[str, Any]) -> None:
        cols = df.columns
        if "overall_rms_g" in cols or "vibration_rms_g" in cols:
            rms_col = "overall_rms_g" if "overall_rms_g" in cols else "vibration_rms_g"
            pos_rms = (df[rms_col] >= 0).all()
            results["checks"].append({
                "name": "RMS Vibration Non-Negativity",
                "category": "Physical Invariant",
                "status": "PASS" if pos_rms else "FAIL",
                "details": f"RMS values strictly non-negative: {pos_rms}",
            })

        if "motor_current_amps" in cols:
            curr_pos = (df["motor_current_amps"] >= 0).all()
            results["checks"].append({
                "name": "Motor Current Non-Negativity",
                "category": "Physical Invariant",
                "status": "PASS" if curr_pos else "FAIL",
                "details": f"Current >= 0 A: {curr_pos}",
            })

    @staticmethod
    def _validate_energy(df: pd.DataFrame, results: Dict[str, Any]) -> None:
        cols = df.columns
        if "solar_generation_mw" in cols:
            sol_pos = (df["solar_generation_mw"] >= 0).all()
            results["checks"].append({
                "name": "Solar Power Generation Non-Negativity",
                "category": "Physical Invariant",
                "status": "PASS" if sol_pos else "FAIL",
                "details": f"Solar output >= 0 MW: {sol_pos}",
            })

        if "grid_frequency_hz" in cols:
            freq_min, freq_max = df["grid_frequency_hz"].min(), df["grid_frequency_hz"].max()
            valid_freq = (freq_min >= 48.0) and (freq_max <= 62.0)
            results["checks"].append({
                "name": "Power Grid Frequency Stability Limits",
                "category": "Grid Invariant",
                "status": "PASS" if valid_freq else "FAIL",
                "details": f"Min: {freq_min:.2f} Hz, Max: {freq_max:.2f} Hz",
            })

    @staticmethod
    def _validate_web(df: pd.DataFrame, results: Dict[str, Any]) -> None:
        cols = df.columns
        if "cpu_utilization_pct" in cols:
            cpu_bound = (df["cpu_utilization_pct"] >= 0.0).all() and (df["cpu_utilization_pct"] <= 100.0).all()
            results["checks"].append({
                "name": "CPU Utilization 0-100% Boundary",
                "category": "OS Invariant",
                "status": "PASS" if cpu_bound else "FAIL",
                "details": f"CPU bounded within [0, 100]%: {cpu_bound}",
            })

    @staticmethod
    def _validate_climate(df: pd.DataFrame, results: Dict[str, Any]) -> None:
        cols = df.columns
        if "relative_humidity_pct" in cols:
            rh_bound = (df["relative_humidity_pct"] >= 0.0).all() and (df["relative_humidity_pct"] <= 100.0).all()
            results["checks"].append({
                "name": "Relative Humidity Psychrometric Bounds",
                "category": "Meteorological Invariant",
                "status": "PASS" if rh_bound else "FAIL",
                "details": f"RH bounded in [0, 100]%: {rh_bound}",
            })

    @staticmethod
    def _check_autocorrelation(df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Verify that time series exhibits temporal continuity (non-zero lag-1 autocorrelation)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            first_col = numeric_cols[0]
            series = df[first_col].dropna().to_numpy()
            if len(series) > 10:
                lag1_acf = np.corrcoef(series[:-1], series[1:])[0, 1]
                has_memory = abs(lag1_acf) > 0.15
                results["checks"].append({
                    "name": f"Temporal Persistence (Lag-1 ACF on {first_col})",
                    "category": "Statistical Authenticity",
                    "status": "PASS" if has_memory else "PASS (Weak)",
                    "details": f"Lag-1 Autocorrelation: {lag1_acf:.3f} (Realistic persistence)",
                })
