"""
synseries.core.evaluator
Safe vectorized formula evaluation for custom variable dependencies in synthetic series.
"""

from typing import Dict, Any
import numpy as np


class SafeFormulaEvaluator:
    """Evaluates mathematical expressions over numpy arrays safely."""

    SAFE_NAMES: Dict[str, Any] = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "clip": np.clip,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "pi": np.pi,
        "e": np.e,
    }

    @classmethod
    def evaluate(
        cls,
        expression: str,
        context: Dict[str, np.ndarray],
        n_steps: int,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Evaluate expression using variable context.
        context: dictionary of variable_name -> np.ndarray
        """
        rng = np.random.default_rng(seed)

        # Environment sandbox
        env = dict(cls.SAFE_NAMES)
        env.update(context)
        env["normal"] = lambda loc=0.0, scale=1.0: rng.normal(loc, scale, n_steps)
        env["uniform"] = lambda low=0.0, high=1.0: rng.uniform(low, high, n_steps)

        # Evaluate safely
        try:
            result = eval(expression, {"__builtins__": {}}, env)
            if isinstance(result, (int, float)):
                return np.full(n_steps, float(result))
            elif isinstance(result, np.ndarray):
                if len(result) != n_steps:
                    raise ValueError(f"Result length {len(result)} != expected {n_steps}")
                return result
            else:
                return np.asarray(result, dtype=float)
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula '{expression}': {e}")
