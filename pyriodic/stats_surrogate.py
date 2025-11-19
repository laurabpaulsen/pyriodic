
import numpy as np
from typing import Callable, Literal, Optional
from .utils import calculate_p_value

def test_against_surrogate(
    stat_fun: Callable,
    observed: np.ndarray,
    surrogate_samples: np.ndarray,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True
    ) -> tuple[float, float, np.ndarray]:

    """
    Test an observed statistic against a distribution of surrogate statistics.

    Parameters
    ----------
    stat_fun : Callable
        A function that computes the statistic of interest. It should take a 1D numpy array as input and return a float (or a tuple where the first element is the statistic).
    observed : np.ndarray
        A 1D array of observed data.
    surrogate_samples : np.ndarray
        A 2D array where each row is a surrogate sample (shape: n_surrogates x sample_length).
    alternative : {'greater', 'less', 'two-sided'}, default='greater'
        Defines the alternative hypothesis for p-value calculation.
    rng : Optional[np.random.Generator], default=None
        A random number generator instance. If None, a new default RNG will be created.
    verbose : bool, default=True
        If True, prints the p-value and statistics.
    """
    

    if rng is None:
        rng = np.random.default_rng()

    # check the shape of surrogate_samples - should be (n_surrogates, sample_length)
    if surrogate_samples.ndim != 2:
        raise ValueError("surrogate_samples should be a 2D array of shape (n_surrogates, sample_length)")
    
    if observed.ndim != 1:
        raise ValueError("observed should be a 1D array of shape (sample_length,)")
    
    if observed.shape[0] != surrogate_samples.shape[1]:
        raise ValueError("observed and surrogate_samples must have the same sample length")

    obs_stat = stat_fun(observed)

    # check if obs_stat is a tuple (some stat functions might return multiple values)
    if isinstance(obs_stat, tuple):
        obs_stat, max_angle = obs_stat
        # Compute obs-vs-null test statistics
        surr_stats = np.apply_along_axis(
            lambda row: stat_fun(row, return_density_at_phase=max_angle, return_angle=False), 1, surrogate_samples
        )
    else:
        # Compute obs-vs-null test statistics
        surr_stats = np.apply_along_axis(stat_fun, 1, surrogate_samples)

    p_val = calculate_p_value(obs_stat, surr_stats, alternative)

    if verbose:
        print(f"p val: {p_val}, observed stat: {obs_stat:.3f}, mean null stat: {np.mean(surr_stats):.3f}")
    
    return (p_val, obs_stat, surr_stats)