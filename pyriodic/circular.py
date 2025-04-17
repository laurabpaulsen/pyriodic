import math
import numpy as np


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

    Attributes
    ----------
    data : ndarray
        The circular data as a NumPy array.
    data_type : str
        Type of circular data (e.g., "angles").
    unit : str
        Unit of measurement, either "radians" or "degrees".


    Methods
    -------
    mean()
    sd()
    convert_to(target_unit)
        Converts the data to a different angular unit ("radians" or "degrees").
    plot(x_ticks="pi")
        Creates a polar plot of the circular data, with density and individual points.
    """

    VALID_UNITS = {"degrees", "radians"}  # hours, years?
    VALID_DATA_TYPES = {"angles"}  # , "directions", "day", "year"}

    UNIT_RANGES = {"radians": 2 * math.pi, "degrees": 360, "hours": 24, "months": 12}

    def __init__(self, data, data_type: str = "angles", unit: str = "radians"):
        unit = unit.lower()
        if unit not in self.VALID_UNITS:
            raise ValueError(
                f"Invalid unit '{unit}'. Must be one of {self.VALID_UNITS}."
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

    # ----- descriptive statistics ------- #
    def mean(self):
        if self.unit == "degrees":
            angles_rad = np.radians(self.data)
        elif self.unit == "radians":
            angles_rad = self.data
        else:
            raise ValueError("unit must be 'radians' or 'degrees'")

        sin_sum = np.sum(np.sin(angles_rad))
        cos_sum = np.sum(np.cos(angles_rad))
        mean_angle = np.arctan2(sin_sum, cos_sum)  # Result is in [-π, π)

        # Normalise output range
        if self.unit == "radians":
            mean_angle = mean_angle % (2 * np.pi)  # range from 0 to 2pi
        elif self.unit == "degrees":
            mean_angle = np.degrees(mean_angle)
            mean_angle = mean_angle % 360

        return mean_angle

    def sd(self):
        """
        Returns the circular standard deviation of the circular data.

        Based on the formula:
            SD = sqrt(-2 * log(R))
        where R is the mean resultant length.
        """
        if self.unit == "degrees":
            angles_rad = np.radians(self.data)
        elif self.unit == "radians":
            angles_rad = self.data
        else:
            raise ValueError("unit must be 'radians' or 'degrees'")

        sin_sum = np.sum(np.sin(angles_rad))
        cos_sum = np.sum(np.cos(angles_rad))
        R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles_rad)

        if R == 0:
            return np.nan  # completely uniform distribution

        circ_sd_rad = np.sqrt(-2 * np.log(R))

        if self.unit == "degrees":
            return np.degrees(circ_sd_rad)
        else:
            return circ_sd_rad

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

    def plot(self, label: str = ""):
        """"""
        from .visualise import PyCircPlot

        plot = PyCircPlot({f"{label}": self})

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
        )
        return summary

    def __repr__(self):
        return f"<Circular(n={len(self.data)}, unit='{self.unit}', type='{self.data_type}')>"
