from .circular import Circular


# ------- one sample --------
import numpy as np
from scipy.stats import kstwobign  # Approx for Kuiper-style tests


def kuiper_test(a: Circular):
    """
    Perform Kuiper's test for uniformity on circular data.

    Parameters
    ----------
    a : Circular
        An instance of the Circular class containing the data to test.
    return_stat : bool
        If True, return the test statistic in addition to the p-value.

    Returns
    -------
    p_value : float
        The p-value for the test.
    stat : float
        The Kuiper test statistic.
    """
    print("WARNING: This function has yet to be tested.")
    if a.unit == "degrees":
        data = np.sort(np.radians(a.data)) % (2 * np.pi)
    else:
        data = np.sort(a.data) % (2 * np.pi)

    n = len(data)
    if n < 2:
        raise ValueError("At least two data points are required.")

    # Normalize data to [0,1)

    data_uniform = data / (2 * np.pi)

    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n

    # Theoretical uniform CDF
    D_plus = np.max(ecdf - data_uniform)
    D_minus = np.max(data_uniform - np.arange(0, n) / n)

    V = D_plus + D_minus

    # Approximate p-value from large-sample distribution
    # (Critical values for Kuiper's statistic are not trivial)
    lambda_v = (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)) * V
    p_value = 2 * np.sum(
        [
            (4 * (k**2) * lambda_v**2 - 1) * np.exp(-2 * (k**2) * lambda_v**2)
            for k in range(1, 1000)
        ]
    )  # Infinite sum approx

    return p_value, V


# ------- group comparison --------


def angular_randomisation_test(a: Circular, b: Circular):
    """
    This function performs the angular randomisation test for homogeneity of 2 groups

    https://ijnaa.semnan.ac.ir/article_5992_e5a258374dedcbb40d792b81bfc94591.pdf


    H0: the samples come from the same population
    HA: the samples do not come from the same population
    """

    # validate the input - do the circular objects have the same unit + datatype?

    pass
