import numpy as np
from typing import Optional, Union

from .utils import dat2rad, rad2dat

from .descriptive_stats import circular_mean, circular_r


class Circular:
    """
    A class for representing and working with circular data (e.g., angles, time-of-day, phase).

    This class supports circular statistics and visualization of data that wraps around a fixed range,
    such as angles (in radians or degrees), phases, hours, or other cyclic measurements. It provides
    basic unit validation, conversion between degrees and radians, and visualisation tools.

    Parameters
    ----------
    data : array-like
        A sequence of numerical values representing circular measurements.
        Values should be in the range appropriate to the specified `unit`.
    data_type : str, optional
        The type of circular data. Currently supported: {"angles"}. Default is "angles".
    unit : str, optional
        The unit of the input data. Must be one of {"radians", "degrees"}. Default is "radians". Assumes radian range to be from 0 to 2pi.
    full_range: int, optional


    Attributes
    ----------
    data : ndarray
        The circular data as a NumPy array.
    unit : str
        Unit of measurement, either "radians" or "degrees".
    """

    def __init__(
        self, data, unit: str = "radians", full_range: Optional[Union[int, None]] = None
    ):

        self.VALID_UNITS = {"degrees", "radians", "hours"}  # hours, years?
        self.UNIT_RANGES = {"radians": 2 * np.pi, "degrees": 360, "hours": 24}

        unit = unit.lower()
        if unit not in self.VALID_UNITS:
            raise ValueError(
                f"Invalid unit '{unit}'. Must be one of {self.VALID_UNITS}."
            )
        self.unit = unit

        if full_range is None:
            try:
                self.full_range = self.UNIT_RANGES[unit]
            except KeyError:
                raise IndexError(
                    f"If unit is not one of {self.VALID_UNITS} you need to specify the full range."
                )
        else:
            self.full_range = full_range

        # check that data is within the specified range
        self._validate_data_matches_range()

        data = np.asarray(data, dtype=float)
        if unit != "radians":
            # convert data to radians
            self.data = dat2rad(data, full_range=self.full_range)
        else:
            self.data = data

    def _validate_data_matches_range(self):
        pass

    # ----- descriptive statistics ------- #
    def mean(self):
        mean = circular_mean(self.data)

        if self.unit != "radians":
            mean = rad2dat(mean, full_range=self.full_range)

        return mean

    def r(self):
        return circular_r(self.data)

    def plot(self, label: str = "", ax=None):
        """"""
        from .visualise import PyCircPlot

        plot = PyCircPlot({f"{label}": self}, ax=ax)

        plot.add_density()
        plot.add_points()

        return plot

    def __str__(self):
        summary = (
            f"Circular Data Object\n"
            f"---------------------\n"
            f"Type:     {self.data_type}\n"
            f"Unit:     {self.unit}\n"
            f"Data:     {self.data[:5]}{'...' if len(self.data) > 5 else ''} "
            f"(n={len(self.data)})\n"
            f"Min value:   {np.min(self.data):.3f}\n"
            f"Max value:   {np.max(self.data):.3f}\n"
            f"Mean:        {self.mean()}"
        )
        return summary

    def __repr__(self):
        return f"<Circular(n={len(self.data)}, unit='{self.unit}', full range = {self.full_range})>"
