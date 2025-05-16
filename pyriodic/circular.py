import numpy as np
from typing import Optional, Union

from .utils import dat2rad, rad2dat

from .desc import circular_mean, circular_r


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
    labels : array-like of str or int, optional
        Optional condition labels or identifiers corresponding to each data point.
    unit : str, optional
        The unit of the input data. Must be one of {"radians", "degrees"}. Default is "radians". Assumes radian range to be from 0 to 2pi.
    full_range: int, optional


    Attributes
    ----------
    data : ndarray
        The circular data as a NumPy array.
    labels : ndarray or list
        Optional labels (e.g., condition tags) for each data point.
    unit : str
        Unit of measurement, either "radians" or "degrees".

    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[Union[list, np.ndarray]] = None,
        unit: str = "radians",
        full_range: Optional[Union[int, None]] = None,
    ):

        self.VALID_UNITS = {"degrees", "radians", "hours"}  # hours, years?
        self.UNIT_RANGES = {"radians": 2 * np.pi + 0.001, "degrees": 360, "hours": 24}

        unit = unit.lower()
        if unit not in self.VALID_UNITS:
            raise ValueError(
                f"Invalid unit '{unit}'. Must be one of {self.VALID_UNITS}."
            )

        self.unit = unit

        if labels is not None and len(labels) != len(data):
            raise ValueError("Length of labels must match length of data.")
        self.labels = labels

        if full_range is None:
            try:
                self.full_range = self.UNIT_RANGES[unit]
            except KeyError:
                raise IndexError(
                    f"If unit is not one of {self.VALID_UNITS} you need to specify the full range."
                )
        else:
            self.full_range = full_range

        data = np.asarray(data, dtype=float)

        # Validate range before conversion
        self._validate_data_matches_range(data, unit, full_range)

        # convert to radians if needed
        if unit != "radians":
            self.data = dat2rad(data, full_range=full_range or self.UNIT_RANGES[unit])
        else:
            self.data = data

    def _validate_data_matches_range(self, data, unit, full_range):
        expected_range = full_range or self.UNIT_RANGES[unit]

        if unit in {"radians", "degrees"}:
            if np.min(data) < 0 or np.max(data) > expected_range:
                raise ValueError(
                    f"Input data values exceed the valid {unit} range (0 to {expected_range}), "
                    f"but unit is set to '{unit}'."
                )

    def mean(self, group_by_label: bool = False):
        if not group_by_label:
            mean = circular_mean(self.data)
            if self.unit != "radians":
                mean = rad2dat(mean, full_range=self.full_range)
            return mean

        if self.labels is None:
            raise ValueError("No labels found for grouping.")

        means = {}
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            mask = self.labels == label
            group_data = self.data[mask]
            mean = circular_mean(group_data)
            if self.unit != "radians":
                mean = rad2dat(mean, full_range=self.full_range)
            means[label] = mean

        return means

    def r(self):
        return circular_r(self.data)

    def plot(self, ax=None, histogram=False, group_by_labels=False):
        """"""
        from .viz import CircPlot

        plot = CircPlot(self, ax=ax, group_by_labels=group_by_labels)

        plot.add_density()
        plot.add_points()
        if histogram is not None:
            plot.add_histogram(data=histogram)

        return plot

    @classmethod
    def from_multiple(cls, circular_objects, labels=None):
        if not circular_objects:
            raise ValueError("No Circular objects provided.")

        units = {c.unit for c in circular_objects}
        if len(units) != 1:
            raise ValueError(f"Inconsistent units in Circular objects: {units}")

        ranges = {c.full_range for c in circular_objects}
        if len(ranges) != 1:
            raise ValueError(f"Inconsistent full_range values: {ranges}")

        # Gather data and generate labels
        all_data = []
        all_labels = []
        if labels is None:
            labels = [f"condition_{i}" for i in range(len(circular_objects))]

        for label, circ in zip(labels, circular_objects):
            all_data.append(circ.data)
            all_labels.extend([label] * len(circ.data))

        all_data = np.concatenate(all_data)
        return cls(
            all_data,
            unit=circular_objects[0].unit,
            full_range=circular_objects[0].full_range,
            labels=np.array(all_labels),
        )

    def __str__(self):
        summary = (
            f"Circular Data Object\n"
            f"---------------------\n"
            f"Unit:       {self.unit}\n"
            f"Full range: {self.full_range}\n"
            f"Data:       {self.data[:5]}{'...' if len(self.data) > 5 else ''} "
            f"(n={len(self.data)})\n"
            f"Min value:  {np.min(self.data):.3f}\n"
            f"Max value:  {np.max(self.data):.3f}\n"
            f"Mean:       {self.mean()}\n"
        )

        if self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            label_lines = "\n".join(
                f"  {label}: {count}" for label, count in zip(unique, counts)
            )
            summary += f"Label distribution:\n{label_lines}"

        return summary

    def __repr__(self):
        return f"<Circular(n={len(self.data)}, unit='{self.unit}', full range = {self.full_range})>"
