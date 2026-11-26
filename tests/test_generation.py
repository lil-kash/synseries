"""
tests/test_generation.py
Unit and integration tests for SynSeries.
"""

import unittest
import numpy as np
import pandas as pd

import synseries as ss
from synseries.core.spec import TimeSeriesDatasetSpec
from synseries.domains.medical import generate_ecg, generate_icu_vitals, generate_cgm
from synseries.domains.finance import generate_ohlcv, generate_portfolio
from synseries.domains.iot_industry import generate_bearing_vibration, generate_predictive_maintenance_run_to_failure
from synseries.domains.energy_grid import generate_smart_grid
from synseries.domains.web_traffic import generate_web_traffic
from synseries.domains.climate import generate_climate_weather


class TestSyntheticTimeSeries(unittest.TestCase):

    def test_medical_ecg(self):
        df = generate_ecg(duration_seconds=3.0, sampling_rate_hz=250, heart_rate_bpm=75.0, seed=42)
        self.assertEqual(len(df), 3 * 250)
        self.assertIn("lead_ii_voltage_mv", df.columns)
        self.assertIn("instant_heart_rate_bpm", df.columns)
        self.assertTrue((df["lead_ii_voltage_mv"] > -3.0).all())
        self.assertTrue((df["lead_ii_voltage_mv"] < 4.0).all())

    def test_medical_icu_vitals(self):
        df = generate_icu_vitals(duration_hours=2.0, condition="septic_shock", seed=42)
        self.assertTrue((df["blood_pressure_systolic_mmhg"] > df["blood_pressure_diastolic_mmhg"]).all())
        self.assertTrue((df["spo2_percent"] <= 100.0).all())
        self.assertTrue((df["spo2_percent"] >= 50.0).all())
        self.assertTrue((df["body_temperature_celsius"] >= 33.0).all())
        self.assertTrue((df["body_temperature_celsius"] <= 43.0).all())

    def test_finance_ohlcv(self):
        df = generate_ohlcv(symbol="AAPL", n_bars=100, seed=42)
        self.assertEqual(len(df), 100)
        self.assertTrue((df["high"] >= df["open"] - 1e-4).all())
        self.assertTrue((df["high"] >= df["close"] - 1e-4).all())
        self.assertTrue((df["low"] <= df["open"] + 1e-4).all())
        self.assertTrue((df["low"] <= df["close"] + 1e-4).all())
        self.assertTrue((df["volume"] >= 0).all())
        self.assertTrue((df["low"] > 0).all())

    def test_iot_bearing_vibration(self):
        df = generate_bearing_vibration(duration_seconds=1.0, sampling_rate_hz=1000, seed=42)
        self.assertEqual(len(df), 1000)
        self.assertIn("accel_x_g", df.columns)
        self.assertIn("accel_y_g", df.columns)
        self.assertIn("accel_z_g", df.columns)
        self.assertTrue((df["overall_rms_g"] >= 0).all())

    def test_smart_grid(self):
        df = generate_smart_grid(duration_days=2.0, seed=42)
        self.assertTrue((df["solar_generation_mw"] >= 0).all())
        self.assertTrue((df["grid_frequency_hz"] >= 48.0).all())
        self.assertTrue((df["grid_frequency_hz"] <= 52.0).all())

    def test_web_traffic(self):
        df = generate_web_traffic(duration_days=2.0, incident_type="memory_leak", seed=42)
        self.assertTrue((df["cpu_utilization_pct"] >= 0).all())
        self.assertTrue((df["cpu_utilization_pct"] <= 100).all())
        self.assertTrue((df["memory_used_mb"] > 0).all())
        self.assertTrue((df["latency_p99_ms"] >= df["latency_p95_ms"]).all())

    def test_prompt_compiler(self):
        spec_med = ss.compile_spec("Generate 24 hours of ICU patient data who has sepsis")
        self.assertEqual(spec_med.domain, "medical")
        self.assertGreater(len(spec_med.variables), 0)

        spec_fin = ss.compile_spec("500 bars of AAPL stock market OHLCV")
        self.assertEqual(spec_fin.domain, "finance")

        spec_iot = ss.compile_spec("3-axis vibration data for motor bearing failure")
        self.assertEqual(spec_iot.domain, "iot")

    def test_yaml_spec_loading(self):
        yaml_content = """name: test_process
domain: custom
start_time: "2026-01-01 00:00:00"
frequency: 1s
n_steps: 100
variables:
  - name: var_a
    base: 10.0
    trend:
      kind: linear
      slope: 0.1
    noise:
      kind: white
      std: 0.5
  - name: var_b
    formula: "2.0 * var_a + 5.0"
"""
        spec = TimeSeriesDatasetSpec.from_yaml(yaml_content)
        df = ss.generate(spec)
        self.assertEqual(len(df), 100)
        self.assertIn("var_a", df.columns)
        self.assertIn("var_b", df.columns)
        # Check formula relationship
        expected_b = 2.0 * df["var_a"] + 5.0
        np.testing.assert_allclose(df["var_b"], expected_b, rtol=1e-5)

    def test_domain_validity_audit(self):
        df_icu = generate_icu_vitals(duration_hours=2.0, seed=42)
        res = ss.validate(df_icu, domain="medical")
        self.assertEqual(res["overall_status"], "PASS")
        self.assertGreaterEqual(res["score_pct"], 85.0)


if __name__ == "__main__":
    unittest.main()
