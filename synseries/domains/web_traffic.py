"""
synseries.domains.web_traffic
Domain-valid Cloud Operations, Web Traffic, and SRE time-series generator:
- Requests per second (RPS) with human diurnal and weekly seasonality
- Latency percentiles (p50, p95, p99) under queueing congestion
- HTTP error rates (5xx, 4xx) and CPU saturation
- Progressive Memory Leak and OOM (Out Of Memory) container crash cycles
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from synseries.core.signal_generators import colored_noise


def generate_web_traffic(
    duration_days: float = 7.0,
    sampling_freq: str = "1min",
    incident_type: Optional[str] = "memory_leak",  # None, 'ddos', 'memory_leak', 'db_outage'
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate realistic Cloud SRE & Web API telemetry:
    - requests_per_sec (RPS)
    - latency_p50_ms
    - latency_p95_ms
    - latency_p99_ms
    - http_5xx_rate_pct
    - cpu_utilization_pct
    - memory_used_mb
    """
    rng = np.random.default_rng(seed)
    n_steps = int(duration_days * 24 * 60)
    timestamps = pd.date_range("2026-03-01 00:00:00", periods=n_steps, freq=sampling_freq)
    t_hours = np.linspace(0, duration_days * 24.0, n_steps)

    # 1. Base Requests Per Second (RPS)
    # Diurnal cycle: low at 04:00, peak at 14:00-16:00
    hour = t_hours % 24.0
    day_of_week = (t_hours / 24.0).astype(int) % 7  # 0=Monday, 5=Sat, 6=Sun
    weekend_factor = np.where(day_of_week >= 5, 0.65, 1.0)

    diurnal = 350.0 + 400.0 * np.sin(2 * np.pi * (hour - 8.0) / 24.0)
    rps = (diurnal * weekend_factor) + colored_noise(n_steps, kind="pink", std=25.0, seed=seed)
    rps = np.maximum(50.0, rps)

    # 2. CPU Utilization: scales with RPS
    cpu = 15.0 + (rps / 1000.0) * 45.0 + colored_noise(n_steps, kind="white", std=2.0, seed=seed + 1 if seed else None)

    # 3. Memory Usage (MB)
    max_memory_mb = 4096.0
    memory_mb = np.full(n_steps, 1200.0) + (rps / 1000.0) * 300.0

    # 4. Latencies (lognormal queueing delays)
    # p50 ~ 25ms, p95 ~ 65ms, p99 ~ 140ms
    latency_p50 = 22.0 + (cpu / 100.0) * 15.0 + rng.normal(0, 1.5, n_steps)
    latency_p95 = latency_p50 * 2.5 + (cpu / 100.0) ** 2 * 40.0 + rng.normal(0, 4.0, n_steps)
    latency_p99 = latency_p95 * 2.2 + (cpu / 100.0) ** 3 * 90.0 + rng.normal(0, 8.0, n_steps)

    # 5. Error Rate (5xx)
    error_rate = 0.05 + 0.1 * (cpu / 100.0) ** 3 + rng.exponential(0.02, n_steps)

    # Incidents injection
    if incident_type == "memory_leak":
        # Linear leak ramp over 3 days, followed by OOM kill restart, then re-ramp
        leak_slope = 0.8  # MB per minute
        leak_acc = 0.0
        for i in range(n_steps):
            leak_acc += leak_slope
            if memory_mb[i] + leak_acc >= max_memory_mb * 0.96:
                # OOM crash!
                memory_mb[i] = 1100.0
                error_rate[i : min(n_steps, i + 5)] += 45.0  # 502 Bad Gateway during restart
                latency_p99[i : min(n_steps, i + 5)] += 1500.0
                leak_acc = 0.0
            else:
                memory_mb[i] += leak_acc

    elif incident_type == "ddos":
        # Sudden 10x traffic spike on Day 3
        ddos_start = int(n_steps * 0.4)
        ddos_duration = 180  # 3 hours
        end = min(n_steps, ddos_start + ddos_duration)
        rps[ddos_start:end] += 6500.0
        cpu[ddos_start:end] = 99.8
        latency_p50[ddos_start:end] += 400.0
        latency_p95[ddos_start:end] += 1200.0
        latency_p99[ddos_start:end] += 3800.0
        error_rate[ddos_start:end] += 32.0

    elif incident_type == "db_outage":
        # High error rate and hanging connections
        outage_start = int(n_steps * 0.5)
        outage_duration = 45  # 45 minutes
        end = min(n_steps, outage_start + outage_duration)
        latency_p99[outage_start:end] += 5000.0  # 5-second connection timeouts
        error_rate[outage_start:end] += 78.0     # 504 Gateway Timeouts

    # Physical bounds clipping
    cpu = np.clip(cpu, 5.0, 100.0)
    memory_mb = np.clip(memory_mb, 500.0, max_memory_mb)
    error_rate = np.clip(error_rate, 0.0, 100.0)
    latency_p50 = np.clip(latency_p50, 5.0, 10000.0)
    latency_p95 = np.maximum(latency_p50 * 1.2, latency_p95)
    latency_p99 = np.maximum(latency_p95 * 1.2, latency_p99)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "requests_per_sec": np.round(rps, 1),
            "cpu_utilization_pct": np.round(cpu, 1),
            "memory_used_mb": np.round(memory_mb, 1),
            "latency_p50_ms": np.round(latency_p50, 1),
            "latency_p95_ms": np.round(latency_p95, 1),
            "latency_p99_ms": np.round(latency_p99, 1),
            "http_5xx_error_pct": np.round(error_rate, 3),
        }
    )
    df = df.set_index("timestamp")
    return df
