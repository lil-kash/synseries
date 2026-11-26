"""
synseries.core.anomalies
Time-series anomaly, event, and regime-shift injection engine.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class AnomalyInjector:
    """Injects point, contextual, collective anomalies and missingness."""

    @staticmethod
    def inject_spike(
        series: np.ndarray,
        indices: List[int],
        magnitude: float,
        relative: bool = False,
    ) -> np.ndarray:
        """Inject single-point spikes or dips."""
        modified = series.copy()
        for idx in indices:
            if 0 <= idx < len(modified):
                if relative:
                    modified[idx] += modified[idx] * magnitude
                else:
                    modified[idx] += magnitude
        return modified

    @staticmethod
    def inject_drift(
        series: np.ndarray,
        start_idx: int,
        duration: int,
        total_slope: float,
    ) -> np.ndarray:
        """Inject progressive sensor drift or gradual degradation."""
        modified = series.copy()
        end_idx = min(len(modified), start_idx + duration)
        if start_idx < len(modified):
            ramp = np.linspace(0, total_slope, end_idx - start_idx)
            modified[start_idx:end_idx] += ramp
        return modified

    @staticmethod
    def inject_regime_shift(
        series: np.ndarray,
        start_idx: int,
        mean_shift: float = 0.0,
        variance_multiplier: float = 1.0,
    ) -> np.ndarray:
        """Shift baseline mean or variance persistently after start_idx."""
        modified = series.copy()
        if 0 <= start_idx < len(modified):
            segment = modified[start_idx:]
            segment_mean = np.mean(segment)
            modified[start_idx:] = (
                segment_mean + mean_shift + (segment - segment_mean) * variance_multiplier
            )
        return modified

    @staticmethod
    def inject_flatline(
        series: np.ndarray,
        start_idx: int,
        duration: int,
        value: Optional[float] = None,
    ) -> np.ndarray:
        """Sensor freeze / flatlining anomaly."""
        modified = series.copy()
        end_idx = min(len(modified), start_idx + duration)
        if start_idx < len(modified):
            freeze_val = series[start_idx] if value is None else value
            modified[start_idx:end_idx] = freeze_val
        return modified

    @staticmethod
    def inject_missingness(
        series: np.ndarray,
        ratio: float = 0.05,
        mechanism: str = "mcar",
        burst_length: int = 5,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Inject missing values (NaN):
        - 'mcar': completely at random points
        - 'burst': consecutive sensor dropouts
        """
        rng = np.random.default_rng(seed)
        modified = series.astype(float).copy()
        n = len(modified)

        if mechanism == "mcar":
            mask = rng.random(n) < ratio
            modified[mask] = np.nan
        elif mechanism == "burst":
            n_bursts = int((n * ratio) / max(1, burst_length))
            starts = rng.integers(0, max(1, n - burst_length), size=n_bursts)
            for s in starts:
                modified[s : s + burst_length] = np.nan
        return modified


def parse_time_offset_to_index(offset: Any, timestamps: pd.DatetimeIndex) -> int:
    """Helper to convert '10m', '2h', or integer to step index."""
    if isinstance(offset, int):
        return offset
    if isinstance(offset, str):
        # E.g. '15m', '2h', '30s'
        delta = pd.to_timedelta(offset)
        target_time = timestamps[0] + delta
        idx = timestamps.searchsorted(target_time)
        return int(min(len(timestamps) - 1, max(0, idx)))
    return 0
