"""
synseries.core.correlations
Multivariate correlation injection using Cholesky decomposition and rank-correlation coupling.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def induce_correlation_matrix(
    data: np.ndarray,
    target_corr: np.ndarray,
) -> np.ndarray:
    """
    Induce a specified target Pearson correlation matrix onto multivariate columns.
    data: shape (N, D) where N is steps, D is dimensions.
    target_corr: shape (D, D) symmetric positive semi-definite matrix.
    """
    n, d = data.shape
    if target_corr.shape != (d, d):
        raise ValueError(f"target_corr shape {target_corr.shape} must match data columns {d}x{d}")

    # Standardize data to zero mean, unit variance
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    standardized = (data - mean) / std

    # Empirical covariance of input data
    r_emp = np.corrcoef(standardized, rowvar=False)

    # Cholesky decomposition of empirical and target
    # Add small epsilon to diagonal for numerical stability
    eps = 1e-7
    r_emp_adj = r_emp + np.eye(d) * eps
    target_adj = target_corr + np.eye(d) * eps

    try:
        l_emp = np.linalg.cholesky(r_emp_adj)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(r_emp_adj)
        eigvals = np.maximum(eigvals, eps)
        l_emp = eigvecs @ np.diag(np.sqrt(eigvals))

    try:
        l_target = np.linalg.cholesky(target_adj)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(target_adj)
        eigvals = np.maximum(eigvals, eps)
        l_target = eigvecs @ np.diag(np.sqrt(eigvals))

    # Decorrelate empirical, then correlate with target
    # X_decorrelated = X @ (L_emp^{-1})^T
    # X_correlated = X_decorrelated @ L_target^T
    inv_l_emp = np.linalg.inv(l_emp)
    transform = inv_l_emp @ l_target

    correlated_data = standardized @ transform

    # Rescale back to original scale and mean
    return correlated_data * std + mean


def build_correlation_matrix_from_dict(
    columns: List[str],
    pairwise_correlations: Dict[str, float],
) -> np.ndarray:
    """
    Construct symmetric positive semi-definite correlation matrix from pairwise specs.
    pairwise_correlations keys: 'colA:colB' -> float [-1.0, 1.0]
    """
    d = len(columns)
    col_idx = {name: i for i, name in enumerate(columns)}
    matrix = np.eye(d)

    for pair, val in pairwise_correlations.items():
        parts = pair.split(":")
        if len(parts) == 2:
            a, b = parts[0].strip(), parts[1].strip()
            if a in col_idx and b in col_idx:
                i, j = col_idx[a], col_idx[b]
                val_clipped = max(-0.99, min(0.99, float(val)))
                matrix[i, j] = val_clipped
                matrix[j, i] = val_clipped

    # Project to nearest positive semi-definite matrix if needed
    eigvals, eigvecs = np.linalg.eigh(matrix)
    if np.any(eigvals < 1e-4):
        eigvals = np.maximum(eigvals, 1e-4)
        matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Renormalize diagonal to 1.0
        d_inv = np.diag(1.0 / np.sqrt(np.diag(matrix)))
        matrix = d_inv @ matrix @ d_inv

    return matrix
