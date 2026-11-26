"""
synseries: High-Fidelity Domain-Valid Synthetic Time-Series Generation Framework.
"""

from synseries.core.spec import (
    TimeSeriesDatasetSpec,
    VariableSpec,
    TrendSpec,
    SeasonalitySpec,
    NoiseSpec,
    AnomalySpec,
)
from synseries.core.engine import TimeSeriesGeneratorEngine
from synseries.nlp.prompt_compiler import PromptCompiler
from synseries.validation.authenticity import DomainAuthenticityValidator
from synseries.validation.reporter import ValidationReporter


def generate(spec_or_prompt, **kwargs):
    """
    Generate synthetic time-series dataset from either:
    1. A natural language string (e.g. 'Generate 24 hours of ICU patient data')
    2. A TimeSeriesDatasetSpec instance
    3. A path to a YAML/JSON spec file
    """
    if isinstance(spec_or_prompt, str):
        if spec_or_prompt.endswith((".yaml", ".yml", ".json")):
            spec = TimeSeriesDatasetSpec.from_file(spec_or_prompt)
        else:
            spec = PromptCompiler.compile(spec_or_prompt)
    elif isinstance(spec_or_prompt, TimeSeriesDatasetSpec):
        spec = spec_or_prompt
    elif isinstance(spec_or_prompt, dict):
        spec = TimeSeriesDatasetSpec.from_dict(spec_or_prompt)
    else:
        raise TypeError(f"Unsupported spec type: {type(spec_or_prompt)}")

    # Apply any kwargs overrides
    for k, v in kwargs.items():
        if hasattr(spec, k):
            setattr(spec, k, v)

    return TimeSeriesGeneratorEngine.generate_from_spec(spec)


def compile_spec(prompt: str) -> TimeSeriesDatasetSpec:
    """Compile natural language prompt into TimeSeriesDatasetSpec."""
    return PromptCompiler.compile(prompt)


def validate(df, domain: str):
    """Validate DataFrame against domain invariants."""
    return DomainAuthenticityValidator.validate(df, domain)


__version__ = "1.0.0"
__all__ = [
    "generate",
    "compile_spec",
    "validate",
    "TimeSeriesDatasetSpec",
    "VariableSpec",
    "TrendSpec",
    "SeasonalitySpec",
    "NoiseSpec",
    "AnomalySpec",
    "TimeSeriesGeneratorEngine",
    "PromptCompiler",
    "DomainAuthenticityValidator",
    "ValidationReporter",
]
