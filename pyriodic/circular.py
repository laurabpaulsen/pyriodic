import math
import numpy as np


class Circular:
    """
    A class for representing and working with circular data (e.g., angles, time-of-day, phase).

    This class supports circular statistics and visualization of data that wraps around a fixed range,
    such as angles (in radians or degrees), phases, hours, or other cyclic measurements. It provides
    basic unit validation, conversion between degrees and radians, and visualization tools.

    Parameters
    ----------
    data : array-like
        A sequence of numerical values representing circular measurements.
        Values should be in the range appropriate to the specified `unit`.
    data_type : str, optional
        The type of circular data. Currently supported: {"angles"}. Default is "angles".
    unit : str, optional
        The unit of the input data. Must be one of {"radians", "degrees"}. Default is "radians".
     zero : str or int, optional
        Defines the zero reference point of the circular data, affecting both its interpretation
        and visualization:

        - If unit is "radians":
            - `zero = 0`  → data is assumed to be in the range **[-π, π]**
            - `zero = "pi"` → data is assumed to be in the range **[0, 2π]**

        This affects tick labeling and density estimation in visualizations.
        Default is 0.

    Attributes
    ----------
    data : ndarray
        The circular data as a NumPy array.
    data_type : str
        Type of circular data (e.g., "angles").
    unit : str
        Unit of measurement, either "radians" or "degrees".
    zero : str or int
        Zero reference point for angular orientation (used in plotting).

    Methods
    -------
    convert_to(target_unit)
        Converts the data to a different angular unit ("radians" or "degrees").
    plot(x_ticks="pi")
        Creates a polar plot of the circular data, with density and individual points.
    """

    VALID_UNITS = {"degrees", "radians"}  # hours, years?
    VALID_DATA_TYPES = {"angles"}  # , "directions", "day", "year"}
    VALID_ZEROS = {"pi", 0}

    UNIT_RANGES = {"radians": 2 * math.pi, "degrees": 360, "hours": 24, "months": 12}

    def __init__(self, data, data_type: str = "angles", unit: str = "radians", zero=0):
        unit = unit.lower()
        if unit not in self.VALID_UNITS:
            raise ValueError(
                f"Invalid unit '{unit}'. Must be one of {self.VALID_UNITS}."
            )

        if zero not in self.VALID_ZEROS:
            raise ValueError(
                f"Invalid zero '{zero}'. Must be one of {self.VALID_ZEROS}."
            )

        data_type = data_type.lower()
        if data_type not in self.VALID_DATA_TYPES:
            raise ValueError(
                (
                    f"Invalid data_type '{data_type}'. Must be one of {self.VALID_DATA_TYPES}."
                )
            )

        # check that data looks valid for the declared unit
        self._validate_data_matches_unit(data, unit)
        self.data = np.asarray(data, dtype=float)
        self.data_type = data_type
        self.unit = unit
        self.zero = zero

    def _validate_data_matches_unit(self, data, unit):
        abs_data = np.abs(data)

        all_within_radian_range = all(
            x <= self.UNIT_RANGES["radians"] + 0.1 for x in abs_data
        )
        any_exceed_radian_range = any(
            x > self.UNIT_RANGES["radians"] + 0.5 for x in abs_data
        )

        if unit == "degrees" and all_within_radian_range:
            raise ValueError(
                "All data values are within the expected radian range (max ~6.28), "
                "but unit is set to 'degrees'. Did you mean 'radians'?"
            )

        if unit == "radians" and any_exceed_radian_range:
            raise ValueError(
                "Some data values exceed the valid radian range (~0–6.28); "
                "they may be in degrees instead."
            )

    def convert_to(self, target_unit):
        if self.unit == target_unit:
            return
        if self.unit == "degrees" and target_unit == "radians":
            self.data = np.deg2rad(self.data)
        elif self.unit == "radians" and target_unit == "degrees":
            self.data = np.rad2deg(self.data)
        else:
            raise NotImplementedError(
                f"Conversion from {self.unit} to {target_unit} not implemented."
            )

        self.unit = target_unit

    def plot(self):
        """"""
        from .visualise import PyCircPlot

        plot = PyCircPlot(self)

        plot.add_density()
        plot.add_points()

        return plot

    def __str__(self):
        summary = (
            f"Circular Data Object\n"
            f"---------------------\n"
            f"Type:     {self.data_type}\n"
            f"Unit:     {self.unit}\n"
            f"Zero ref:    {self.zero}\n"
            f"Data:     {self.data[:5]}{'...' if len(self.data) > 5 else ''} "
            f"(n={len(self.data)})\n"
            f"Min value:   {np.min(self.data):.3f}\n"
            f"Max value:   {np.max(self.data):.3f}\n"
        )
        return summary

    def __repr__(self):
        return f"<Circular(n={len(self.data)}, unit='{self.unit}', type='{self.data_type}')>"
