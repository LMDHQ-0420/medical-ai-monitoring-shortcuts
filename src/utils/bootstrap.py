import numpy as np
from typing import Callable


def bootstrap_ci(
    func: Callable,
    *arrays: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a scalar metric function.

    Args:
        func: metric function accepting the same positional arrays, returns float
        *arrays: arrays to resample together (must have same first dimension)
        n_bootstrap: number of bootstrap iterations
        alpha: significance level (0.05 → 95% CI)
        seed: random seed

    Returns:
        (lower, upper) confidence interval bounds
    """
    rng = np.random.default_rng(seed)
    n = arrays[0].shape[0]
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        resampled = [arr[idx] for arr in arrays]
        try:
            val = func(*resampled)
            if np.isfinite(val):
                stats.append(val)
        except Exception:
            continue
    if len(stats) < 10:
        return (float("nan"), float("nan"))
    stats = np.array(stats)
    lower = float(np.percentile(stats, 100 * alpha / 2))
    upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lower, upper
