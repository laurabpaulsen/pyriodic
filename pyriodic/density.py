import numpy as np
from math import pi

def vonmises_kde(data, kappa, min_x=0, max_x=2 * pi, n_bins=100):
    from scipy.special import i0

    bins = np.linspace(min_x, max_x, n_bins)
    x = np.linspace(min_x, max_x, n_bins)
    # integrate vonmises kernels
    kde = np.exp(kappa * np.cos(x[:, None] - data[None, :])).sum(1) / (
        2 * np.pi * i0(kappa)
    )
    kde /= np.trapz(kde, x=bins)
    return bins, kde