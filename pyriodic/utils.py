from typing import Union, Literal
import numpy as np


def dat2rad(data: Union[np.ndarray, float, int], full_range: Union[float, int] = 360):

    return 2 * np.pi * data / full_range


def rad2dat(rad: Union[np.ndarray, float, int], full_range: Union[float, int] = 360):

    return full_range * rad / (2 * np.pi)


def calculate_p_value(
    observed_stat: float,
    null_distribution: np.ndarray,
    alternative: Literal["greater", "less", "two-sided"] = "greater"
) -> float:
    """
    Compute the p-value for a given observed statistic against a null distribution.

    Parameters
    ----------
    observed_stat : float
        The observed test statistic.

    null_distribution : np.ndarray
        Array of test statistics computed under the null hypothesis.

    alternative : {'greater', 'less', 'two-sided'}, default='greater'
        Defines the alternative hypothesis:
        - 'greater': test if observed > null
        - 'less': test if observed < null
        - 'two-sided': test if observed is different from null (absolute deviation)

    Returns
    -------
    p_value : float
        The computed p-value.
    """
    null_distribution = np.asarray(null_distribution)
    n = len(null_distribution)

    if alternative == "greater":
        p = (np.sum(null_distribution >= observed_stat) + 1) / (n + 1)
    elif alternative == "less":
        p = (np.sum(null_distribution <= observed_stat) + 1) / (n + 1)
    elif alternative == "two-sided":
        null_mean = np.mean(null_distribution)
        p = (np.sum(np.abs(null_distribution - null_mean) >= np.abs(observed_stat - null_mean)) + 1) / (n + 1)
    else:
        raise ValueError("Invalid alternative. Choose from 'greater', 'less', or 'two-sided'.")

    return p
