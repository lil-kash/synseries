"""
synseries.nlp.prompt_compiler
Compiles natural language prompt requests into valid, domain-accurate TimeSeriesDatasetSpec objects.
Supports standard domain presets as well as fully customized arbitrary user specifications.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from synseries.core.spec import (
    TimeSeriesDatasetSpec,
    VariableSpec,
    TrendSpec,
    SeasonalitySpec,
    NoiseSpec,
    AnomalySpec,
)


class PromptCompiler:
    """Translates user natural language instructions into concrete TimeSeriesDatasetSpec instances."""

    @classmethod
    def compile(cls, prompt: str) -> TimeSeriesDatasetSpec:
        """Parse natural language prompt into a full, valid TimeSeriesDatasetSpec."""
        p_lower = prompt.lower()

        # Check for explicit custom multi-channel indicators first
        has_custom_indicators = any(
            k in p_lower for k in [
                "channels:", "variables:", "sensors:", "named ", "channels named",
                "custom sensor", "formula", "between ", "starting at"
            ]
        )

        if has_custom_indicators and not any(w in p_lower for w in ["ecg", "cgm", "ohlcv"]):
            return cls._compile_dynamic_custom(p_lower, prompt)

        # 1. Medical Domain
        if any(w in p_lower for w in ["medical", "health", "ecg", "cardiac", "patient", "icu", "vitals", "glucose", "cgm", "doctor", "hospital"]):
            return cls._compile_medical(p_lower, prompt)

        # 2. Finance / Trading Domain
        elif any(w in p_lower for w in ["finance", "stock", "market", "ohlcv", "trading", "crypto", "btc", "portfolio", "forex", "candle", "shares"]):
            return cls._compile_finance(p_lower, prompt)

        # 3. IoT / Industrial Domain
        elif any(w in p_lower for w in ["iot", "bearing", "vibration", "motor", "predictive maintenance", "machine", "factory", "rpm", "wear", "rul"]):
            return cls._compile_iot(p_lower, prompt)

        # 4. Energy & Smart Grid Domain
        elif any(w in p_lower for w in ["energy", "solar", "wind", "grid", "electricity", "frequency", "power", "megawatt", "photovoltaic", "turbine"]):
            return cls._compile_energy(p_lower, prompt)

        # 5. Cloud & Web Traffic Domain
        elif any(w in p_lower for w in ["web", "traffic", "server", "cloud", "latency", "rps", "devops", "sre", "cpu", "memory leak", "5xx", "requests"]):
            return cls._compile_web(p_lower, prompt)

        # 6. Climate & Meteorology Domain
        elif any(w in p_lower for w in ["climate", "weather", "temperature", "humidity", "rain", "meteorology", "solar radiation", "barometric", "wind speed"]):
            return cls._compile_climate(p_lower, prompt)

        # 7. Fully Customized Arbitrary Synthesis
        else:
            return cls._compile_dynamic_custom(p_lower, prompt)

    @classmethod
    def _extract_time_parameters(cls, p: str) -> Tuple[Optional[str], Optional[int], str]:
        """Extract duration, n_steps, and frequency from text."""
        duration = None
        n_steps = None
        frequency = "1s"

        # Match n_steps first: e.g. "500 points", "1000 steps", "200 samples", "100 bars"
        steps_match = re.search(r"(\d+)\s*(?:points|steps|samples|bars|rows|readings)\b", p)
        if steps_match:
            n_steps = int(steps_match.group(1))

        # Match duration: require explicit time units with word boundary to avoid matching "with", "where"
        dur_match = re.search(r"(\d+(?:\.\d+)?)\s*(seconds?|sec|minutes?|min|hours?|hr|days?|weeks?)\b", p)
        if dur_match and not steps_match:
            val, unit = dur_match.group(1), dur_match.group(2)
            if unit.startswith("s"):
                duration = f"{int(float(val))}s"
            elif unit.startswith("min"):
                duration = f"{int(float(val))}min"
            elif unit.startswith("h"):
                duration = f"{int(float(val))}h"
            elif unit.startswith("d"):
                duration = f"{int(float(val))}d"
            elif unit.startswith("w"):
                duration = f"{int(float(val))}w"

        # Match frequency: e.g. "100ms", "5s", "1min", "1h", "daily", "hourly"
        freq_match = re.search(r"(\d+(?:\.\d+)?)\s*(ms|s|min|h|d)\s*(?:interval|step|frequency|rate)?\b", p)
        if freq_match:
            frequency = f"{freq_match.group(1)}{freq_match.group(2)}"
        elif "hourly" in p:
            frequency = "1h"
        elif "daily" in p:
            frequency = "1d"
        elif "per second" in p or "every second" in p:
            frequency = "1s"

        if duration is None and n_steps is None:
            n_steps = 1000

        return duration, n_steps, frequency

    @classmethod
    def _compile_dynamic_custom(cls, p: str, orig: str) -> TimeSeriesDatasetSpec:
        """
        Parses customized requests, extracting:
        - Named channels / variables
        - Baseline levels, trends, cycles, and noise
        - Custom formulas
        - Targeted anomalies (spikes, drops, drifts, flatlines)
        """
        duration, n_steps, frequency = cls._extract_time_parameters(p)

        # 1. Identify Variables / Channels to create
        variables: List[VariableSpec] = []
        correlations: Dict[str, float] = {}
        anomalies: List[AnomalySpec] = []

        # Check for multi-clause specifications (e.g. "sensor_a starting at 100, sensor_b with noise 0.2")
        cleaned_p = re.sub(r"^(?:generate|synthesize|create)\s*(?:\d+\s*)?(?:channels|variables|sensors)?[\s:=]*", "", p, flags=re.I)
        raw_clauses = [c.strip() for c in re.split(r"[,;]|\band\b", cleaned_p) if c.strip()]
        
        clause_vars: List[Tuple[str, str]] = []  # (var_name, clause_text)
        stopwords = {
            "a", "an", "the", "with", "and", "of", "in", "at", "for", "from",
            "points", "steps", "samples", "bars", "low", "high", "sudden", "daily",
            "hourly", "upward", "downward", "linear", "random", "clean", "noisy",
            "retail", "store", "custom", "sensor", "sensors", "dataset", "data"
        }
        for c in raw_clauses:
            m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)", c)
            if m:
                vname = m.group(1).lower()
                if vname not in stopwords and not vname.isdigit():
                    clause_vars.append((vname, c))

        known_keywords = [
            ("temperature", "°C", 25.0, 5.0, 50.0),
            ("temp", "°C", 25.0, 5.0, 50.0),
            ("pressure", "bar", 3.5, 0.5, 10.0),
            ("voltage", "V", 230.0, 200.0, 250.0),
            ("current", "A", 12.0, 0.0, 40.0),
            ("humidity", "%", 55.0, 10.0, 95.0),
            ("flow_rate", "L/min", 45.0, 0.0, 120.0),
            ("vibration", "g", 0.8, 0.0, 15.0),
            ("rpm", "rpm", 1800.0, 0.0, 5000.0),
            ("speed", "km/h", 65.0, 0.0, 180.0),
            ("power", "kW", 15.0, 0.0, 100.0),
            ("cpu", "%", 35.0, 0.0, 100.0),
            ("memory", "MB", 2048.0, 256.0, 8192.0),
            ("latency", "ms", 45.0, 5.0, 2000.0),
            ("throughput", "req/s", 500.0, 0.0, 10000.0),
            ("sales", "units", 150.0, 0.0, 2000.0),
            ("demand", "MW", 80.0, 10.0, 300.0),
        ]

        candidate_configs: List[Tuple[str, str]] = []  # (name, context_str)

        matched_kws = [(kw, p) for kw, unit, b, mn, mx in known_keywords if re.search(rf"\b{kw}\b", p)]

        if len(clause_vars) >= 2:
            candidate_configs = clause_vars[:8]
        elif matched_kws:
            candidate_configs = matched_kws[:6]
        elif clause_vars:
            candidate_configs = clause_vars[:8]
        else:
            candidate_configs = [("signal_alpha", p), ("signal_beta", p)]

        # Parse base values and parameters per candidate variable
        for var_name, var_context in candidate_configs:
            base_val = 50.0
            min_val = None
            max_val = None
            unit = ""

            for kw, u, b, mn, mx in known_keywords:
                if kw in var_name:
                    unit = u
                    base_val = b
                    min_val = mn
                    max_val = mx
                    break

            # Search base in local clause context first, then global
            base_match = re.search(r"(?:starting at|base|around|from|level)\s*([+-]?\d+(?:\.\d+)?)", var_context)
            if not base_match:
                base_match = re.search(rf"{var_name}[^\.,]*?(?:starting at|base|around|from|level)\s*([+-]?\d+(?:\.\d+)?)", p)
            if base_match:
                base_val = float(base_match.group(1))
                min_val = None
                max_val = None

            # Detect trend (gentle slope relative to series horizon)
            trend_spec = TrendSpec(kind="none")
            ctx_lower = var_context.lower()
            rel_slope = 0.002 * (abs(base_val) if base_val != 0 else 1.0)
            if any(w in ctx_lower for w in ["upward", "rising", "increasing", "growth"]):
                trend_spec = TrendSpec(kind="linear", slope=rel_slope)
            elif any(w in ctx_lower for w in ["downward", "falling", "decreasing", "decay"]):
                trend_spec = TrendSpec(kind="linear", slope=-rel_slope)
            elif "exponential" in ctx_lower:
                trend_spec = TrendSpec(kind="exponential", growth_rate=0.0005, intercept=base_val)

            # Detect seasonality / cycles
            seasonality_list = []
            if any(w in ctx_lower for w in ["cycle", "periodic", "seasonal", "wave", "sinusoid", "daily", "oscillation"]):
                period = 50.0
                if "daily" in ctx_lower or "diurnal" in ctx_lower:
                    period = 24.0
                amp = max(1.0, abs(base_val) * 0.15)
                seasonality_list.append(SeasonalitySpec(period=period, amplitude=amp, waveform="sine"))

            # Detect noise level
            noise_std = max(0.5, abs(base_val) * 0.04)
            noise_match = re.search(r"noise\s*(?:std|level)?\s*([0-9]+(?:\.[0-9]+)?)", ctx_lower)
            if not noise_match:
                noise_match = re.search(r"noise\s*(?:std|level)?\s*([0-9]+(?:\.[0-9]+)?)", p)
            if noise_match:
                noise_std = float(noise_match.group(1))
            elif "low noise" in ctx_lower or "clean" in ctx_lower:
                noise_std = max(0.1, abs(base_val) * 0.01)
            elif "high noise" in ctx_lower or "noisy" in ctx_lower:
                noise_std = max(2.0, abs(base_val) * 0.15)

            # Check if this variable is derived from a formula (e.g. "signal_b = 2 * signal_a + 5")
            formula = None
            formula_match = re.search(rf"{var_name}\s*=\s*([a-zA-Z0-9_\.\s\+\-\*\/\(\)]+)", p)
            if formula_match:
                formula = formula_match.group(1).strip()

            var_spec = VariableSpec(
                name=var_name,
                unit=unit,
                base=base_val,
                min_val=min_val,
                max_val=max_val,
                trend=trend_spec,
                seasonality=seasonality_list,
                noise=NoiseSpec(kind="pink" if "pink" in p else "white", std=noise_std),
                formula=formula,
            )
            variables.append(var_spec)

        # 2. Detect Anomalies & Injections
        target_name = variables[0].name if variables else "all"

        # Spikes / Surges
        if any(w in p for w in ["spike", "surge", "impulse"]):
            mag_match = re.search(r"(?:spike|surge)[^\.,]*?([0-9]+(?:\.[0-9]+)?)", p)
            mag = float(mag_match.group(1)) if mag_match else (variables[0].base * 0.5 if variables else 25.0)
            step_pos = int((n_steps or 1000) * 0.45)
            anomalies.append(AnomalySpec(type="spike", target=target_name, start=step_pos, magnitude=mag))

        # Sudden Drops / Dips
        if any(w in p for w in ["drop", "dip", "crash", "plunge"]):
            mag_match = re.search(r"(?:drop|dip|crash)[^\.,]*?([0-9]+(?:\.[0-9]+)?)", p)
            mag = -float(mag_match.group(1)) if mag_match else (-variables[0].base * 0.4 if variables else -20.0)
            step_pos = int((n_steps or 1000) * 0.5)
            anomalies.append(AnomalySpec(type="spike", target=target_name, start=step_pos, magnitude=mag))

        # Drift / Leak
        if any(w in p for w in ["drift", "leak", "ramp"]):
            step_pos = int((n_steps or 1000) * 0.3)
            dur_steps = int((n_steps or 1000) * 0.4)
            drift_mag = variables[0].base * 0.3 if variables else 15.0
            anomalies.append(AnomalySpec(type="drift", target=target_name, start=step_pos, duration=dur_steps, magnitude=drift_mag))

        # Flatline / Sensor Freeze
        if any(w in p for w in ["flatline", "freeze", "stuck"]):
            step_pos = int((n_steps or 1000) * 0.6)
            dur_steps = int((n_steps or 1000) * 0.2)
            anomalies.append(AnomalySpec(type="flatline", target=target_name, start=step_pos, duration=dur_steps))

        # Missing values / Dropouts
        if any(w in p for w in ["missing", "dropout", "gap", "nan"]):
            anomalies.append(AnomalySpec(type="missingness", target=target_name, start=0, params={"ratio": 0.05, "mechanism": "mcar"}))

        # 3. Inter-channel correlation if multiple variables
        if len(variables) >= 2 and not variables[1].formula:
            correlations[f"{variables[0].name}:{variables[1].name}"] = 0.70

        return TimeSeriesDatasetSpec(
            name="custom_synthesized_telemetry",
            domain="custom",
            description=f"Synthesized from custom specification: '{orig}'",
            frequency=frequency,
            duration=duration,
            n_steps=n_steps if duration is None else None,
            variables=variables,
            correlations=correlations,
            anomalies=anomalies,
        )

    @classmethod
    def _compile_medical(cls, p: str, orig: str) -> TimeSeriesDatasetSpec:
        duration, n_steps, freq = cls._extract_time_parameters(p)

        if "ecg" in p or "cardiac" in p or "heartbeat" in p:
            arrh = None
            if "pvc" in p or "premature" in p:
                arrh = "pvc"
            elif "bradycardia" in p:
                arrh = "bradycardia"
            elif "tachycardia" in p:
                arrh = "tachycardia"
            elif "st elevation" in p or "infarction" in p or "attack" in p:
                arrh = "st_elevation"

            dur = 10.0
            dur_match = re.search(r"(\d+(?:\.\d+)?)\s*(s|sec|seconds)", p)
            if dur_match:
                dur = float(dur_match.group(1))

            hr_bpm = 72.0
            hr_match = re.search(r"(?:heart rate|hr|bpm)\s*[:=]?\s*(\d+)", p)
            if hr_match:
                hr_bpm = float(hr_match.group(1))

            return TimeSeriesDatasetSpec(
                name="clinical_ecg_lead_ii",
                domain="medical",
                description=f"Synthetic Lead II ECG waveform generated from: '{orig}'",
                frequency="4ms",
                duration=f"{dur}s",
                metadata={"type": "ecg", "duration_seconds": dur, "heart_rate_bpm": hr_bpm, "arrhythmia": arrh},
                variables=[
                    VariableSpec(name="lead_ii_voltage_mv", unit="mV", description="Lead II voltage", min_val=-2.5, max_val=3.0),
                    VariableSpec(name="instant_heart_rate_bpm", unit="bpm", base=hr_bpm, description="Instantaneous heart rate", min_val=30.0, max_val=220.0),
                ]
            )

        elif "glucose" in p or "cgm" in p or "diabetic" in p:
            ptype = "type_1_diabetic" if "type 1" in p else ("type_2_diabetic" if "type 2" in p else "healthy")
            days = 3.0
            days_match = re.search(r"(\d+(?:\.\d+)?)\s*(d|day|days)", p)
            if days_match:
                days = float(days_match.group(1))

            return TimeSeriesDatasetSpec(
                name="clinical_cgm_telemetry",
                domain="medical",
                description=f"Continuous Glucose Monitoring dataset generated from: '{orig}'",
                frequency="5min",
                duration=f"{days}d",
                metadata={"type": "cgm", "duration_days": days, "patient_type": ptype},
                variables=[
                    VariableSpec(name="glucose_mg_dl", unit="mg/dL", description="Interstitial glucose level", min_val=40.0, max_val=450.0),
                    VariableSpec(name="glycemic_zone", unit="category", description="Clinical glycemic state classification"),
                ]
            )

        else:
            # ICU vitals
            cond = "septic_shock" if ("sepsis" in p or "shock" in p) else ("hypoxia" if "hypoxia" in p else "normal")
            dur_h = 12.0
            dur_match = re.search(r"(\d+(?:\.\d+)?)\s*(h|hr|hours)", p)
            if dur_match:
                dur_h = float(dur_match.group(1))

            return TimeSeriesDatasetSpec(
                name="icu_patient_telemetry",
                domain="medical",
                description=f"ICU multi-vital physiological monitoring generated from: '{orig}'",
                frequency="5s",
                duration=f"{dur_h}h",
                metadata={"type": "icu", "duration_hours": dur_h, "condition": cond},
                variables=[
                    VariableSpec(name="heart_rate_bpm", unit="bpm", base=74.0, min_val=30.0, max_val=220.0),
                    VariableSpec(name="blood_pressure_systolic_mmhg", unit="mmHg", base=118.0, min_val=40.0, max_val=240.0),
                    VariableSpec(name="blood_pressure_diastolic_mmhg", unit="mmHg", base=76.0, min_val=25.0, max_val=150.0),
                    VariableSpec(name="spo2_percent", unit="%", base=98.0, min_val=60.0, max_val=100.0),
                    VariableSpec(name="respiration_rate_bpm", unit="breaths/min", base=15.0, min_val=4.0, max_val=50.0),
                    VariableSpec(name="body_temperature_celsius", unit="°C", base=36.8, min_val=33.0, max_val=42.5),
                ],
                correlations={"heart_rate_bpm:blood_pressure_systolic_mmhg": 0.25, "blood_pressure_systolic_mmhg:blood_pressure_diastolic_mmhg": 0.88}
            )

    @classmethod
    def _compile_finance(cls, p: str, orig: str) -> TimeSeriesDatasetSpec:
        if "portfolio" in p or "assets" in p or "basket" in p:
            days = 365
            d_match = re.search(r"(\d+)\s*(days?|d)", p)
            if d_match:
                days = int(d_match.group(1))
            return TimeSeriesDatasetSpec(
                name="quantitative_portfolio_prices",
                domain="finance",
                description=f"Correlated multi-asset portfolio generated from: '{orig}'",
                frequency="1B",
                duration=f"{days}d",
                metadata={"type": "portfolio", "n_days": days, "symbols": ["SPY", "QQQ", "GLD", "TLT", "BTC"]},
                variables=[
                    VariableSpec(name="SPY_price", unit="USD", description="S&P 500 Index ETF"),
                    VariableSpec(name="QQQ_price", unit="USD", description="Nasdaq 100 ETF"),
                    VariableSpec(name="GLD_price", unit="USD", description="SPDR Gold Trust"),
                    VariableSpec(name="TLT_price", unit="USD", description="20+ Year Treasury Bond"),
                    VariableSpec(name="BTC_price", unit="USD", description="Bitcoin Cryptocurrency"),
                ]
            )
        else:
            sym = "AAPL"
            sym_match = re.search(r"\b([a-z]{2,5})\s+(stock|share|crypto|price)", p)
            if sym_match:
                sym = sym_match.group(1).upper()

            bars = 500
            bars_match = re.search(r"(\d+)\s*(bars|candles|steps)", p)
            if bars_match:
                bars = int(bars_match.group(1))

            start_price = 150.0
            price_match = re.search(r"(?:price|starting at|at)\s*(\d+(?:\.\d+)?)", p)
            if price_match:
                start_price = float(price_match.group(1))

            return TimeSeriesDatasetSpec(
                name=f"{sym.lower()}_market_ohlcv",
                domain="finance",
                description=f"OHLCV market candlestick series generated from: '{orig}'",
                frequency="1h",
                n_steps=bars,
                metadata={"type": "ohlcv", "symbol": sym, "n_bars": bars, "initial_price": start_price},
                variables=[
                    VariableSpec(name="open", unit="USD", base=start_price),
                    VariableSpec(name="high", unit="USD", base=start_price * 1.01),
                    VariableSpec(name="low", unit="USD", base=start_price * 0.99),
                    VariableSpec(name="close", unit="USD", base=start_price),
                    VariableSpec(name="volume", unit="shares", base=500000),
                    VariableSpec(name="vwap", unit="USD", base=start_price),
                ]
            )

    @classmethod
    def _compile_iot(cls, p: str, orig: str) -> TimeSeriesDatasetSpec:
        if "vibration" in p or "bearing" in p or "accelerometer" in p:
            defect = "outer_race" if "outer" in p else ("unbalance" if "unbalance" in p else "outer_race")
            dur = 5.0
            dur_match = re.search(r"(\d+(?:\.\d+)?)\s*(s|sec|seconds)", p)
            if dur_match:
                dur = float(dur_match.group(1))

            return TimeSeriesDatasetSpec(
                name="industrial_bearing_vibration",
                domain="iot",
                description=f"High-frequency 3-axis vibration telemetry generated from: '{orig}'",
                frequency="500us",
                duration=f"{dur}s",
                metadata={"type": "vibration", "duration_seconds": dur, "defect_type": defect},
                variables=[
                    VariableSpec(name="accel_x_g", unit="g", description="X-axis radial acceleration"),
                    VariableSpec(name="accel_y_g", unit="g", description="Y-axis radial acceleration"),
                    VariableSpec(name="accel_z_g", unit="g", description="Z-axis axial acceleration"),
                    VariableSpec(name="overall_rms_g", unit="g", description="Overall Root Mean Square"),
                ]
            )
        else:
            days = 30
            d_match = re.search(r"(\d+)\s*(days?|d)", p)
            if d_match:
                days = int(d_match.group(1))

            return TimeSeriesDatasetSpec(
                name="predictive_maintenance_run_to_failure",
                domain="iot",
                description=f"Full lifecycle run-to-failure degradation dataset generated from: '{orig}'",
                frequency="10min",
                duration=f"{days}d",
                metadata={"type": "rul", "n_days": days},
                variables=[
                    VariableSpec(name="vibration_rms_g", unit="g", min_val=0.1, max_val=10.0),
                    VariableSpec(name="bearing_temperature_celsius", unit="°C", min_val=20.0, max_val=140.0),
                    VariableSpec(name="motor_current_amps", unit="A", min_val=0.0, max_val=120.0),
                    VariableSpec(name="acoustic_emission_db", unit="dB", min_val=20.0, max_val=120.0),
                    VariableSpec(name="remaining_useful_life_hours", unit="hours", min_val=0.0),
                    VariableSpec(name="health_status", unit="status", description="HEALTHY / DEGRADED / CRITICAL"),
                ]
            )

    @classmethod
    def _compile_energy(cls, p: str, orig: str) -> TimeSeriesDatasetSpec:
        days = 7.0
        d_match = re.search(r"(\d+(?:\.\d+)?)\s*(days?|d)", p)
        if d_match:
            days = float(d_match.group(1))

        return TimeSeriesDatasetSpec(
            name="smart_microgrid_telemetry",
            domain="energy",
            description=f"Smart microgrid solar, wind, and frequency telemetry generated from: '{orig}'",
            frequency="15min",
            duration=f"{days}d",
            metadata={"type": "smart_grid", "duration_days": days, "has_grid_trip": "trip" in p or "blackout" in p or "failure" in p},
            variables=[
                VariableSpec(name="solar_generation_mw", unit="MW", min_val=0.0, max_val=100.0),
                VariableSpec(name="wind_generation_mw", unit="MW", min_val=0.0, max_val=80.0),
                VariableSpec(name="total_renewable_mw", unit="MW", min_val=0.0),
                VariableSpec(name="demand_load_mw", unit="MW", min_val=10.0),
                VariableSpec(name="net_power_balance_mw", unit="MW"),
                VariableSpec(name="grid_frequency_hz", unit="Hz", base=50.0, min_val=48.5, max_val=51.5),
            ]
        )

    @classmethod
    def _compile_web(cls, p: str, orig: str) -> TimeSeriesDatasetSpec:
        inc = "memory_leak"
        if "ddos" in p:
            inc = "ddos"
        elif "db" in p or "database" in p or "outage" in p:
            inc = "db_outage"

        days = 7.0
        d_match = re.search(r"(\d+(?:\.\d+)?)\s*(days?|d)", p)
        if d_match:
            days = float(d_match.group(1))

        return TimeSeriesDatasetSpec(
            name="cloud_devops_apm_telemetry",
            domain="web_traffic",
            description=f"Cloud operations and APM telemetry generated from: '{orig}'",
            frequency="1min",
            duration=f"{days}d",
            metadata={"type": "web_traffic", "duration_days": days, "incident_type": inc},
            variables=[
                VariableSpec(name="requests_per_sec", unit="req/s", min_val=0.0),
                VariableSpec(name="cpu_utilization_pct", unit="%", min_val=0.0, max_val=100.0),
                VariableSpec(name="memory_used_mb", unit="MB", min_val=0.0, max_val=4096.0),
                VariableSpec(name="latency_p50_ms", unit="ms", min_val=0.0),
                VariableSpec(name="latency_p95_ms", unit="ms", min_val=0.0),
                VariableSpec(name="latency_p99_ms", unit="ms", min_val=0.0),
                VariableSpec(name="http_5xx_error_pct", unit="%", min_val=0.0, max_val=100.0),
            ]
        )

    @classmethod
    def _compile_climate(cls, p: str, orig: str) -> TimeSeriesDatasetSpec:
        season = "summer" if "summer" in p or "hot" in p else ("winter" if "winter" in p or "cold" in p else "monsoon")
        days = 14.0
        d_match = re.search(r"(\d+(?:\.\d+)?)\s*(days?|d)", p)
        if d_match:
            days = float(d_match.group(1))

        return TimeSeriesDatasetSpec(
            name="climate_meteorology_sensors",
            domain="climate",
            description=f"Meteorological and atmospheric multi-sensor series generated from: '{orig}'",
            frequency="1h",
            duration=f"{days}d",
            metadata={"type": "climate", "duration_days": days, "season": season},
            variables=[
                VariableSpec(name="solar_radiation_w_m2", unit="W/m²", min_val=0.0),
                VariableSpec(name="air_temperature_celsius", unit="°C"),
                VariableSpec(name="relative_humidity_pct", unit="%", min_val=0.0, max_val=100.0),
                VariableSpec(name="barometric_pressure_hpa", unit="hPa"),
                VariableSpec(name="wind_speed_m_s", unit="m/s", min_val=0.0),
            ]
        )
