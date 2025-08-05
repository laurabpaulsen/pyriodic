import numpy as np
from typing import Callable, Optional, Literal, Union
import inspect
from .utils import calculate_p_value
from scipy.stats import mannwhitneyu



def ecdf(data: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute the Empirical Cumulative Distribution Function (ECDF) of the given data at specified points.

    Parameters
    ----------
    data : np.ndarray
        The input sample data.
    points : np.ndarray
        The points at which to evaluate the ECDF.

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
    phase_pool: Union[np.ndarray, Literal["uniform"]] = "uniform",
    n_null: int = 1000,
    n_permutations:int = 1000,
    stat_fun: Optional[Callable] = None,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    rng: Optional[np.random.Generator] = None,
    n_bins: int = 30,
    verbose: bool = True,
    return_null_samples=False,
    return_obs_and_null_stats=False
) -> tuple:
    """
    Compares an observed circular phase distribution to a null distribution
    using a Mann–Whitney U-based permutation test on test statistic samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    if stat_fun is None:
        stat_fun = watson_u2  # You must define/import this separately

    # Determine if stat_fun accepts n_bins
    sig = inspect.signature(stat_fun)
    accepts_n_bins = "n_bins" in sig.parameters

    def compute_stat(x, y):
        return stat_fun(x, y, n_bins=n_bins) if accepts_n_bins else stat_fun(x, y)

    n_events = len(observed)

    # Generate null samples
    if isinstance(phase_pool, str) and phase_pool == "uniform":
        phase_pool = np.linspace(0, 2 * np.pi, len(observed), endpoint=False)
    elif not isinstance(phase_pool, np.ndarray):
        raise ValueError("phase_pool must be a numpy array or 'uniform'.")

    null_samples = [
        rng.choice(phase_pool, size=n_events, replace=False) for _ in range(n_null)
    ]

    # Compute obs-vs-null test statistics
    obs_vs_null = np.array([compute_stat(observed, null) for null in null_samples])

    # Compute null-vs-null test statistics
    null_vs_null = np.array([
        compute_stat(*rng.choice(null_samples, size=2, replace=False))
        for _ in range(n_null)
    ])

    # Observed U statistic
    obs_stat = mannwhitneyu(obs_vs_null, null_vs_null, alternative="two-sided").statistic

    # Permutation test on labels
    group_labels = np.array([1] * len(obs_vs_null) + [0] * len(null_vs_null))
    all_stats = np.concatenate([obs_vs_null, null_vs_null])
    perm_stats = []

    for _ in range(n_permutations):
        permuted = rng.permutation(group_labels)
        group1 = all_stats[permuted == 1]
        group0 = all_stats[permuted == 0]
        stat = mannwhitneyu(group1, group0, alternative="two-sided").statistic
        perm_stats.append(stat)

    perm_stats = np.array(perm_stats)
    p_val = calculate_p_value(obs_stat, perm_stats, alternative)

    if verbose:
        print(f"p val: {p_val}, observed stat: {obs_stat}:.1f, mean null stat: {np.mean(perm_stats):.1f}")

    
    result = [obs_stat, p_val]
    if return_null_samples:
        result.append(np.array(null_samples))

    if return_obs_and_null_stats:
        result.append(obs_vs_null)
        result.append(null_vs_null)

    return tuple(result)


def permutation_test_between_samples(
    sample1: np.ndarray,
    sample2: np.ndarray,
    stat_fun: Optional[Callable] = None,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
    n_bins: int = 30,
    verbose: bool = True,
    return_null_distribution: bool = False,
) -> tuple:
    """
    NOTE: Documentation to be added
    """

    if rng is None:
        rng = np.random.default_rng()

    if stat_fun is None:
        stat_fun = watson_u2

    sig = inspect.signature(stat_fun)
    accepts_n_bins = "n_bins" in sig.parameters

    def compute_stat(x, y):
        return stat_fun(x, y, n_bins=n_bins) if accepts_n_bins else stat_fun(x, y)

    # Compute observed statistic
    observed_stat = compute_stat(sample1, sample2)

    # Combine data and generate permutations
    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    null_distribution = []

    for _ in range(n_permutations):
        permuted = rng.permutation(combined)
        perm1 = permuted[:n1]
        perm2 = permuted[n1:]
        stat = compute_stat(perm1, perm2)
        null_distribution.append(stat)

    p_value = calculate_p_value(observed_stat, np.array(null_distribution), alternative)

    if verbose:
        print(f"Observed statistic = {observed_stat:.3f}, p = {p_value:.4f}")

    if return_null_distribution:
        return observed_stat, p_value, null_distribution
    else:
        return observed_stat, p_value


def paired_permutation_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    return_null_distribution: bool = False,
) -> tuple:
    """
    Performs a permutation test on paired samples (e.g., group-level stat comparison).
    """
    if rng is None:
        rng = np.random.default_rng()

    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    if sample1.shape != sample2.shape:
        raise ValueError("Samples must have the same shape (paired data).")

    observed_diff = np.mean(sample1 - sample2)
    null_distribution = []

    for _ in range(n_permutations):
        # Randomly flip signs of differences
        signs = rng.choice([-1, 1], size=sample1.shape)
        perm_diff = np.mean(signs * (sample1 - sample2))
        null_distribution.append(perm_diff)

    p_value = calculate_p_value(observed_diff, np.array(null_distribution), alternative)

    if verbose:
        print(f"Observed mean difference = {observed_diff:.3f}, p = {p_value:.4f}")

    if return_null_distribution:
        return observed_diff, p_value, null_distribution
    else:
        return observed_diff, p_value


def peak_to_peak_modulation(
    phases: np.ndarray,
    values: np.ndarray,
    n_bins: int = 8,
    min_bin_count: int = 10
) -> float:
    """
    Computes the peak-to-peak modulation (max - min) of a continuous variable
    binned by circular phase.

    Parameters
    ----------
    phases : np.ndarray
        1D array of phase values (in radians, from 0 to 2π).
    values : np.ndarray
        1D array of continuous values (e.g., reaction times), same length as `phases`.
    n_bins : int, default=8
        Number of phase bins to divide the cycle into.
    min_bin_count : int, default=10
        Minimum number of samples required in each bin. If any bin has fewer, an error is raised.

    Returns
    -------
    modulation : float
        Difference between the max and min average value across phase bins.
    """
    if len(phases) != len(values):
        raise ValueError("phases and values must be the same length.")

    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_indices = np.digitize(phases, bin_edges, right=False) - 1
    bin_indices[bin_indices == n_bins] = 0  # handle phase == 2π

    binned_means = []
    for i in range(n_bins):
        bin_vals = values[bin_indices == i]
        if len(bin_vals) < min_bin_count:
            raise ValueError(
                f"Bin {i} has only {len(bin_vals)} samples (minimum is {min_bin_count})."
            )
        binned_means.append(np.mean(bin_vals))

    modulation = np.max(binned_means) - np.min(binned_means)
    return modulation




from typing import Optional, Callable, Literal

def permutation_test_phase_modulation(
    phases: np.ndarray,
    values: np.ndarray,
    stat_fun: Optional[Callable] = None,
    n_null: int = 1000,
    n_bins: int = 8,
    min_bin_count: int = 10,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    return_null_distribution: bool = False
) -> tuple:
    """
    Perform a permutation test to assess whether a linear variable is modulated by circular phase.

    Parameters
    ----------
    phases : np.ndarray
        1D array of circular phase values (in radians).
    values : np.ndarray
        1D array of scalar values (e.g., RTs) to test for phase-dependent modulation.
    stat_fun : Callable, optional
        Function that computes a modulation index given (phases, values).
        Defaults to peak-to-peak modulation across phase bins.
    n_null : int, default=1000
        Number of permutations to build the null distribution.
    n_bins : int, default=8
        Number of bins to use when binning phase (passed to `stat_fun`).
    min_bin_count : int, default=10
        Minimum number of samples per bin (passed to `stat_fun`).
    alternative : {"greater", "less", "two-sided"}, default="greater"
        The alternative hypothesis to test.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    verbose : bool, default=True
        If True, prints the observed statistic and p-value.
    return_null_distribution : bool, default=False
        If True, also returns the null distribution of test statistics.

    Returns
    -------
    obs_stat : float
        The observed modulation index.
    p_val : float
        P-value of the observed statistic under the null distribution.
    null_dist : np.ndarray, optional
        The null distribution (if `return_null_distribution` is True).
    """
    # 1. Set RNG
    if rng is None:
        rng = np.random.default_rng()

    # default stat function
    if stat_fun is None:
        stat_fun = peak_to_peak_modulation  # <-- the one we just wrote

    # compute observed statistic
    obs_stat = stat_fun(phases, values, n_bins=n_bins, min_bin_count=min_bin_count)

    # generate null distribution
    null = []
    for _ in range(n_null):
        permuted_values = rng.permutation(values)
        null_stat = stat_fun(phases, permuted_values, n_bins=n_bins, min_bin_count=min_bin_count)
        null.append(null_stat)
    null = np.array(null)

    # 5. Compute p-value
    p_val = calculate_p_value(obs_stat, null, alternative=alternative)

    if verbose:
        print(f"Observed statistic = {obs_stat:.3f}, p = {p_val:.4f}")

    if return_null_distribution:
        return obs_stat, p_val, null

    return obs_stat, p_val
