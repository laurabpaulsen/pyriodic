import numpy as np
from scipy.special import i0

def vonmises_kde(data, kappa, min_x=0, max_x=2*np.pi, n_bins=100):
    """
    Vectorized von Mises kernel density estimate for circular data.

    Parameters
    ----------
    data : np.ndarray
        Circular data in radians.
    kappa : float
        Concentration parameter (higher = narrower peaks).
    min_x : float
        Minimum value of the evaluation grid (default 0).
    max_x : float
        Maximum value of the evaluation grid (default 2*pi).
    n_bins : int
        Number of points in the evaluation grid.

    Returns
    -------
    x : np.ndarray
        Evaluation points.
    kde : np.ndarray
        Estimated density at each point.
    """
    x = np.linspace(min_x, max_x, n_bins)
    # Compute vectorized von Mises kernels
    kde = np.exp(kappa * np.cos(x[:, None] - data[None, :])).sum(axis=1) / (2 * np.pi * i0(kappa))
    # Normalize
    kde /= np.trapz(kde, x=x)
    return x, kde