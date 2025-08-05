import numpy as np
from typing import Optional, Union


def circular_mean(rad: np.ndarray, wrap_to_2pi: bool = True):
    """
    Compute the mean angle from a list of angles in radians.

    Parameters:
    ------------
    rad : np.ndarray
        Array of angles in radians.

    wrap_to_2pi : bool, default=True
        If True, wraps result to range [0, 2π]. If False, result is in [-π, π].

    Returns:
    --------
    mean : float
        The circular mean angle.

    """
    sin_sum = np.sum(np.sin(rad))
    cos_sum = np.sum(np.cos(rad))

    mean = np.arctan2(sin_sum, cos_sum)

    if wrap_to_2pi and mean < 0:
        mean += 2 * np.pi

    return mean



def circular_median(rad: np.ndarray):
    """
    Compute the median angle from a list of angles in radians.
    
    Parameters:
    ------------
    rad : np.ndarray
        Array of angles in radians.
    Returns:
    --------
    median : float

        The circular median angle.

    """
    sorted_rad = np.sort(rad)
    n = len(sorted_rad)
    
    if n % 2 == 1:
        return sorted_rad[n // 2]
    else:
        mid1 = sorted_rad[n // 2 - 1]
        mid2 = sorted_rad[n // 2]
        return circular_mean(np.array([mid1, mid2]))


def circular_r(rad: np.ndarray):
    r"""
    Compute the length of the mean resultant vector $r$, a measure of circular concentration.

    .. math::

        r = \sqrt{\bar{C}^2 + \bar{S}^2}
    """
    sin_sum = np.sum(np.sin(rad))
    cos_sum = np.sum(np.cos(rad))
    n = rad.size

    r = np.sqrt(sin_sum**2 + cos_sum**2) / n
    return r


def angular_deviation(rad: np.ndarray):
    raise NotImplementedError


def circular_standard_deviation(rad: np.ndarray):
    raise NotImplementedError
