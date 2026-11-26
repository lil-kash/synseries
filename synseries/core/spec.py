"""
synseries.core.spec
Data models, schema specification, and YAML/JSON serialization for synthetic time-series.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
import yaml
import json


@dataclass
class NoiseSpec:
    kind: str = "white"  # white, pink, brown
    std: float = 1.0


@dataclass
class SeasonalitySpec:
    period: Union[float, str] = 24.0  # steps or duration string (e.g. "24h")
    amplitude: float = 1.0
    waveform: str = "sine"  # sine, cosine, sawtooth, square, diurnal
    phase: float = 0.0


@dataclass
class TrendSpec:
    kind: str = "none"  # none, linear, polynomial, exponential, logistic
    slope: float = 0.0
    intercept: float = 0.0
    growth_rate: float = 0.001
    carrying_capacity: float = 100.0


@dataclass
class VariableSpec:
    name: str
    unit: str = ""
    description: str = ""
    base: float = 0.0
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    trend: TrendSpec = field(default_factory=TrendSpec)
    seasonality: List[SeasonalitySpec] = field(default_factory=list)
    noise: NoiseSpec = field(default_factory=NoiseSpec)
    process_type: Optional[str] = None  # ou, merton, ar, ecg, etc.
    process_params: Dict[str, Any] = field(default_factory=dict)
    formula: Optional[str] = None  # e.g., "1.2 * varA - 0.5 * varB"


@dataclass
class AnomalySpec:
    type: str  # spike, drift, regime_shift, flatline, missingness, event
    target: str  # variable name or 'all'
    start: Union[int, str] = 0  # index or offset string e.g. "2h"
    duration: Union[int, str] = 1
    magnitude: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesDatasetSpec:
    name: str
    domain: str = "custom"
    description: str = ""
    start_time: str = "2026-01-01 00:00:00"
    frequency: str = "1s"
    duration: Optional[str] = None
    n_steps: Optional[int] = 1000
    variables: List[VariableSpec] = field(default_factory=list)
    correlations: Dict[str, float] = field(default_factory=dict)  # "varA:varB": 0.75
    anomalies: List[AnomalySpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to standard python dictionary."""
        return asdict(self)

    def to_yaml(self) -> str:
        """Serialize specification to clean YAML format."""
        return yaml.dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def to_json(self, indent: int = 2) -> str:
        """Serialize specification to JSON format."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeriesDatasetSpec":
        """Reconstruct spec object from dictionary."""
        d = data.copy()
        raw_vars = d.pop("variables", [])
        variables = []
        for v in raw_vars:
            trend_data = v.get("trend", {})
            if isinstance(trend_data, dict):
                trend = TrendSpec(**trend_data)
            else:
                trend = TrendSpec()

            noise_data = v.get("noise", {})
            if isinstance(noise_data, dict):
                noise = NoiseSpec(**noise_data)
            else:
                noise = NoiseSpec()

            raw_seas = v.get("seasonality", [])
            seasonality = [
                SeasonalitySpec(**s) if isinstance(s, dict) else s for s in raw_seas
            ]

            var = VariableSpec(
                name=v.get("name", "var"),
                unit=v.get("unit", ""),
                description=v.get("description", ""),
                base=v.get("base", 0.0),
                min_val=v.get("min_val"),
                max_val=v.get("max_val"),
                trend=trend,
                seasonality=seasonality,
                noise=noise,
                process_type=v.get("process_type"),
                process_params=v.get("process_params", {}),
                formula=v.get("formula"),
            )
            variables.append(var)

        raw_anomalies = d.pop("anomalies", [])
        anomalies = [
            AnomalySpec(**a) if isinstance(a, dict) else a for a in raw_anomalies
        ]

        return cls(variables=variables, anomalies=anomalies, **d)

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "TimeSeriesDatasetSpec":
        """Load spec from YAML string."""
        data = yaml.safe_load(yaml_content)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, filepath: str) -> "TimeSeriesDatasetSpec":
        """Load spec from a YAML or JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            elif filepath.endswith(".json"):
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        return cls.from_dict(data)
