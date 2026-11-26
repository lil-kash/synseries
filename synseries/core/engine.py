"""
synseries.core.engine
Master generation engine compiling declarative TimeSeriesDatasetSpec into concrete datasets.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from synseries.core.spec import (
    TimeSeriesDatasetSpec,
    VariableSpec,
    AnomalySpec,
)
from synseries.core.signal_generators import (
    generate_timestamps,
    trend as generate_trend,
    seasonality as generate_seasonality,
    colored_noise,
    ornstein_uhlenbeck,
    merton_jump_diffusion,
    autoregressive_process,
)
from synseries.core.correlations import (
    induce_correlation_matrix,
    build_correlation_matrix_from_dict,
)
from synseries.core.anomalies import AnomalyInjector, parse_time_offset_to_index
from synseries.core.evaluator import SafeFormulaEvaluator

# Domain generator imports
import synseries.domains.medical as medical_dom
import synseries.domains.finance as finance_dom
import synseries.domains.iot_industry as iot_dom
import synseries.domains.energy_grid as energy_dom
import synseries.domains.web_traffic as web_dom
import synseries.domains.climate as climate_dom


class TimeSeriesGeneratorEngine:
    """Master engine orchestrating procedural and domain-specific time series generation."""

    @classmethod
    def generate_from_spec(cls, spec: TimeSeriesDatasetSpec) -> pd.DataFrame:
        """Compile and execute a TimeSeriesDatasetSpec into a pandas DataFrame."""
        # 1. Check if domain preset shortcut applies
        domain_df = cls._try_domain_shortcut(spec)
        if domain_df is not None:
            return domain_df

        # 2. Determine number of steps and timestamps
        freq = spec.frequency or "1s"
        start = spec.start_time or "2026-01-01 00:00:00"

        if spec.duration is not None:
            total_delta = pd.to_timedelta(spec.duration)
            end_time = pd.Timestamp(start) + total_delta
            timestamps = generate_timestamps(start=start, end=end_time, freq=freq)
            n_steps = len(timestamps)
        else:
            n_steps = spec.n_steps or 1000
            timestamps = generate_timestamps(start=start, n_steps=n_steps, freq=freq)

        rng = np.random.default_rng(spec.random_seed)
        t_steps = np.arange(n_steps)

        # 3. Generate primary variables (non-formula)
        data: Dict[str, np.ndarray] = {}
        pending_formulas: List[VariableSpec] = []

        for var_idx, var in enumerate(spec.variables):
            if var.formula:
                pending_formulas.append(var)
                continue

            var_seed = None if spec.random_seed is None else spec.random_seed + var_idx * 17
            series = np.full(n_steps, float(var.base))

            # Stochastic process
            if var.process_type == "ou":
                theta = var.process_params.get("theta", 0.15)
                mu = var.process_params.get("mu", var.base)
                sigma = var.process_params.get("sigma", 1.0)
                series = ornstein_uhlenbeck(n_steps, theta=theta, mu=mu, sigma=sigma, seed=var_seed)
            elif var.process_type == "merton":
                s0 = var.process_params.get("s0", var.base)
                mu = var.process_params.get("mu", 0.08)
                sigma = var.process_params.get("sigma", 0.20)
                series = merton_jump_diffusion(n_steps, s0=s0, mu=mu, sigma=sigma, seed=var_seed)
            elif var.process_type == "ar":
                coeffs = var.process_params.get("coeffs", [0.7])
                series = autoregressive_process(n_steps, ar_coeffs=coeffs, intercept=var.base, seed=var_seed)
            else:
                # Add Trend
                if var.trend and var.trend.kind != "none":
                    series += generate_trend(
                        t_steps,
                        kind=var.trend.kind,
                        slope=var.trend.slope,
                        intercept=var.trend.intercept,
                        growth_rate=var.trend.growth_rate,
                        carrying_capacity=var.trend.carrying_capacity,
                    )

                # Add Seasonality
                if var.seasonality:
                    periods = []
                    amplitudes = []
                    phases = []
                    waveforms = []
                    for s in var.seasonality:
                        p_val = s.period
                        if isinstance(p_val, str):
                            p_delta = pd.to_timedelta(p_val)
                            p_step = p_delta / pd.to_timedelta(freq)
                            periods.append(float(p_step))
                        else:
                            periods.append(float(p_val))
                        amplitudes.append(s.amplitude)
                        phases.append(s.phase)
                        waveforms.append(s.waveform)

                    series += generate_seasonality(t_steps, periods, amplitudes, phases, waveforms)

                # Add Noise
                if var.noise and var.noise.std > 0:
                    series += colored_noise(n_steps, kind=var.noise.kind, std=var.noise.std, seed=var_seed)

            data[var.name] = series

        # 4. Induce multivariate correlations if specified
        if spec.correlations and len(data) > 1:
            numeric_cols = [k for k in data.keys()]
            corr_matrix = build_correlation_matrix_from_dict(numeric_cols, spec.correlations)
            mat_data = np.column_stack([data[k] for k in numeric_cols])
            correlated_mat = induce_correlation_matrix(mat_data, corr_matrix)
            for i, col in enumerate(numeric_cols):
                data[col] = correlated_mat[:, i]

        # 5. Evaluate dependent formula variables
        for var in pending_formulas:
            var_seed = None if spec.random_seed is None else spec.random_seed + 99
            data[var.name] = SafeFormulaEvaluator.evaluate(
                expression=var.formula,
                context=data,
                n_steps=n_steps,
                seed=var_seed or 42,
            )

        # 6. Apply Anomalies & Events
        for anom in spec.anomalies:
            target_cols = list(data.keys()) if anom.target == "all" else [anom.target]
            start_idx = parse_time_offset_to_index(anom.start, timestamps)
            duration_idx = parse_time_offset_to_index(anom.duration, timestamps)
            duration_idx = max(1, duration_idx)

            for col in target_cols:
                if col not in data:
                    continue

                if anom.type == "spike":
                    data[col] = AnomalyInjector.inject_spike(
                        data[col], [start_idx], magnitude=anom.magnitude
                    )
                elif anom.type == "drift":
                    data[col] = AnomalyInjector.inject_drift(
                        data[col], start_idx, duration_idx, total_slope=anom.magnitude
                    )
                elif anom.type == "regime_shift":
                    var_mult = anom.params.get("variance_multiplier", 1.0)
                    data[col] = AnomalyInjector.inject_regime_shift(
                        data[col], start_idx, mean_shift=anom.magnitude, variance_multiplier=var_mult
                    )
                elif anom.type == "flatline":
                    val = anom.params.get("value")
                    data[col] = AnomalyInjector.inject_flatline(data[col], start_idx, duration_idx, value=val)
                elif anom.type == "missingness":
                    ratio = anom.params.get("ratio", 0.05)
                    mechanism = anom.params.get("mechanism", "mcar")
                    data[col] = AnomalyInjector.inject_missingness(data[col], ratio=ratio, mechanism=mechanism)

        # 7. Apply Physical Bounds Clipping (min_val, max_val)
        for var in spec.variables:
            if var.name in data:
                low = -np.inf if var.min_val is None else var.min_val
                high = np.inf if var.max_val is None else var.max_val
                data[var.name] = np.clip(data[var.name], low, high)

        # 8. Assemble into clean DataFrame
        df = pd.DataFrame(data, index=timestamps)
        df.index.name = "timestamp"

        # Attach metadata to DataFrame attrs
        df.attrs["spec_name"] = spec.name
        df.attrs["domain"] = spec.domain
        df.attrs["metadata"] = spec.metadata

        return df

    @classmethod
    def _try_domain_shortcut(cls, spec: TimeSeriesDatasetSpec) -> Optional[pd.DataFrame]:
        """Detect if domain preset shortcut should be invoked directly."""
        domain = (spec.domain or "").lower()
        meta = spec.metadata or {}
        seed = spec.random_seed or 42

        # Check for direct preset flags in metadata
        sub_type = meta.get("type", "").lower()
        if not sub_type:
            return None

        if domain == "medical":
            if "ecg" in sub_type or "cardiac" in sub_type:
                dur = meta.get("duration_seconds", 10.0)
                hr = meta.get("heart_rate_bpm", 72.0)
                arrh = meta.get("arrhythmia", None)
                return medical_dom.generate_ecg(duration_seconds=dur, heart_rate_bpm=hr, arrhythmia=arrh, seed=seed)
            elif "cgm" in sub_type or "glucose" in sub_type:
                days = meta.get("duration_days", 3.0)
                ptype = meta.get("patient_type", "type_1_diabetic")
                return medical_dom.generate_cgm(duration_days=days, patient_type=ptype, seed=seed)
            else:
                # Default medical is ICU vitals
                dur_h = meta.get("duration_hours", 12.0)
                cond = meta.get("condition", "normal")
                return medical_dom.generate_icu_vitals(duration_hours=dur_h, condition=cond, seed=seed)

        elif domain in ["finance", "trading", "quant"]:
            if "portfolio" in sub_type:
                symbols = meta.get("symbols", ["SPY", "QQQ", "GLD", "TLT", "BTC"])
                days = meta.get("n_days", 365)
                return finance_dom.generate_portfolio(symbols=symbols, n_days=days, seed=seed)
            else:
                bars = spec.n_steps or meta.get("n_bars", 500)
                sym = meta.get("symbol", "AAPL")
                return finance_dom.generate_ohlcv(symbol=sym, n_bars=bars, seed=seed)

        elif domain in ["iot", "manufacturing", "industry"]:
            if "vibration" in sub_type or "bearing" in sub_type:
                dur = meta.get("duration_seconds", 5.0)
                defect = meta.get("defect_type", "outer_race")
                return iot_dom.generate_bearing_vibration(duration_seconds=dur, defect_type=defect, seed=seed)
            else:
                days = meta.get("n_days", 30)
                return iot_dom.generate_predictive_maintenance_run_to_failure(n_days=days, seed=seed)

        elif domain in ["energy", "smart_grid", "grid"]:
            days = meta.get("duration_days", 7.0)
            trip = meta.get("has_grid_trip", True)
            return energy_dom.generate_smart_grid(duration_days=days, has_grid_trip=trip, seed=seed)

        elif domain in ["web_traffic", "devops", "cloud", "sre"]:
            days = meta.get("duration_days", 7.0)
            inc = meta.get("incident_type", "memory_leak")
            return web_dom.generate_web_traffic(duration_days=days, incident_type=inc, seed=seed)

        elif domain in ["climate", "weather", "meteorology"]:
            days = meta.get("duration_days", 14.0)
            season = meta.get("season", "summer")
            return climate_dom.generate_climate_weather(duration_days=days, season=season, seed=seed)

        return None
