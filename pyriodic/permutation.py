import numpy as np
from typing import Callable, Optional, List
import inspect


def ecdf(data: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute the Empirical Cumulative Distribution Function (ECDF) of the given data at specified points.

    Parameters:
    - data (np.ndarray): The input sample data.
    - points (np.ndarray): The points at which to evaluate the ECDF.

    Returns:
    - np.ndarray: ECDF values evaluated at each point in `points`.
    """
    return np.array([np.sum(data <= x) for x in points]) / len(data)


def watson_u2(sample1: np.ndarray, sample2: np.ndarray, n_bins: int = 30) -> float:
    """
    Compute the Watson U² statistic between two circular samples.

    This is a non-parametric test statistic used to compare two samples of circular data,
    based on the squared differences between their ECDFs.

    Parameters:
    - sample1 (np.ndarray): First sample of circular data (values should be in [0, 2π]).
    - sample2 (np.ndarray): Second sample of circular data (values should be in [0, 2π]).
    - n_bins (int): Number of points to evaluate the ECDFs on the circle.

    Returns:
    - float: The Watson U² statistic.
    """
    points = np.linspace(0, 2 * np.pi, n_bins)
    F1 = ecdf(sample1, points)
    F2 = ecdf(sample2, points)
    return np.sum((F1 - F2) ** 2)


def permutation_test_against_null(
    observed: np.ndarray,
    null_samples: np.ndarray,
    n_null: int = 1000,
    stat_fun: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
    n_bins: int = 30,
    verbose: bool = True,
) -> tuple[float, List[float], float]:
    """
    Perform a permutation test to evaluate whether observed circular data
    significantly differs from a null distribution of permuted samples.

    This function compares an observed dataset (e.g., participant data) against
    a set of permuted versions of that data using a circular test statistic
    (default: Watson U²). It also estimates a null distribution by comparing
    pairs of permuted samples to one another.

    Parameters
    ----------
    observed : np.ndarray
        1D array of observed angular data in radians (values should be in [0, 2π]).

    null_samples : np.ndarray
        2D array of permuted samples (shape: [n_permutations, n_points]).
        Each row is a permutation of the observed data under the null hypothesis.

    n_null : int, default=1000
        Number of permutation-to-permutation comparisons used to generate the null distribution.

    stat_fun : Callable, optional
        Function used to compute the test statistic. Must accept at least two 1D arrays.
        If it accepts an additional `n_bins` argument, it will be passed automatically.
        Defaults to `watson_u2`.

    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a default generator is created.

    n_bins : int, default=30
        Number of evaluation points for ECDF-based statistics (used only if `stat_fun` accepts it).

    verbose : bool, default=True
        If True, prints the observed statistic and corresponding p-value.

    Returns
    -------
    obs : float
        The mean statistic between the observed data and all permuted samples.

    null : list of float
        The null distribution: statistics computed between randomly selected pairs of permuted samples.

    p_val : float
        The p-value, calculated as the proportion of null statistics greater than or equal to `obs`.

    Notes
    -----
    This test assumes that the permuted samples represent valid null-distribution surrogates
    and that the test statistic is sensitive to distributional differences on the circle.
    """
    if rng is None:
        rng = np.random.default_rng()

    if stat_fun is None:
        stat_fun = watson_u2

    # Determine if stat_fun accepts n_bins
    sig = inspect.signature(stat_fun)
    accepts_n_bins = "n_bins" in sig.parameters

    def compute_stat(x, y):
        return stat_fun(x, y, n_bins=n_bins) if accepts_n_bins else stat_fun(x, y)

    # Compute observed-to-null distances
    obs_vs_null = [compute_stat(observed, perm) for perm in null_samples]
    obs = np.mean(obs_vs_null)

    # Build null distribution
    null = []
    for _ in range(n_null):
        p1, p2 = rng.choice(null_samples, size=2, replace=False)
        null.append(compute_stat(p1, p2))

    # Compute p-value
    p_val = (np.sum(np.array(null) >= obs) + 1) / (len(null) + 1)

    if verbose:
        print(f"Observed statistic = {obs:.3f}, p = {p_val:.4f}")

    return obs, null, p_val
