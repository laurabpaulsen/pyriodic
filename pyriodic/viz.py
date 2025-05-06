import matplotlib.pyplot as plt
import numpy as np
from .circular import Circular
from math import pi
from typing import Union, Optional
from .desc import circular_mean, circular_r


DEFAULT_COLOUR = "forestgreen"


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


class CircPlot:
    def __init__(
        self,
        circ: Circular,
        group_by_labels: bool = True,
        colours=None,
        fig_size=(6, 6),
        dpi=300,
        ax=None,
        ylim=None,
    ):
        self.circ = circ
        self.group_by_labels = group_by_labels

        if ax is None:
            self.fig, self.ax = plt.subplots(
                figsize=fig_size, dpi=dpi, subplot_kw={"projection": "polar"}
            )
        else:
            # Ensure it's a polar axis
            if ax.name != "polar":
                raise ValueError("Provided axis must be a polar projection")
            self.ax = ax
            self.fig = ax.figure

        self.prepare_ax(ylim=ylim)

        if colours:
            self.colours = [colours] if colours.type() == "str" else colours
        else:
            # Automatically pick colors from matplotlib's default cycle
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            color_cycle = prop_cycle.by_key()["color"]
            if group_by_labels:
                if self.circ.labels is None:
                    raise ValueError(
                        "Can only group by labels if labels are present in the circular object"
                    )
                self.colours = color_cycle[: len(np.unique(self.circ.labels))]

    def prepare_ax(self, ylim):
        # Remove radial ticks
        self.ax.set_yticklabels([])
        self.ax.set_yticks([])

        # Optional: remove radial gridlines
        self.ax.yaxis.grid(False)

        # Optional: remove theta gridlines
        # self.ax.xaxis.grid(False)

        # Optional: set custom radius limit
        if ylim:
            self.ax.set_ylim(*ylim)

        # Set theta offset

        # self.ax.set_theta_offset(zero_location)

        # Direction of theta
        self.ax.set_theta_direction(-1)

    def add_points(self, grouped: Optional[bool] = None, **kwargs):
        """
        Plot circular data points on the polar axis.

        Parameters
        ----------
        grouped : bool, optional
            If None, uses the `self.group_by_labels` setting.
            If True, plots each label separately.
            If False, plots all data points together, ignoring labels.
        kwargs : dict
            Additional keyword arguments passed to `ax.scatter`, such as `s`, `alpha`, or `marker`.

        """
        if grouped is None:
            grouped = self.group_by_labels

        if grouped:
            if self.circ.labels is None:
                raise ValueError(
                    "Cannot group by labels: no labels present in the Circular object."
                )

            unique_labels = np.unique(self.circ.labels)
            for idx, label in enumerate(unique_labels):
                values = self.circ.data[self.circ.labels == label]
                self.ax.scatter(
                    values,
                    [0.5] * len(values),
                    color=self.colours[idx % len(self.colours)],
                    label=label,
                    **kwargs,
                )
        else:
            self.ax.scatter(
                self.circ.data,
                [0.5] * len(self.circ.data),
                color=DEFAULT_COLOUR,
                label="all",
                **kwargs,
            )

    def add_density(
        self, kappa=20, n_bins=500, grouped: Optional[bool] = None, **kwargs
    ):
        """
        Add a circular density estimate using Von Mises KDE.

        Parameters
        ----------
        kappa : float
            Concentration parameter of the Von Mises distribution (higher = sharper).
        n_bins : int
            Number of angular bins to evaluate the density on.
        grouped : bool, optional
            If None, uses `self.group_by_labels`.
            If True, plots a density curve for each label separately.
            If False, plots a single joint density curve for all data.
        kwargs : dict
            Additional keyword arguments passed to `ax.plot` (e.g., linewidth, linestyle, alpha).
        """
        if grouped is None:
            grouped = self.group_by_labels

        if grouped:
            if self.circ.labels is None:
                raise ValueError(
                    "Cannot group by labels: no labels present in the Circular object."
                )

            unique_labels = np.unique(self.circ.labels)
            for idx, label in enumerate(unique_labels):
                values = self.circ.data[self.circ.labels == label]
                if len(values) == 0:
                    continue

                min_x, max_x = min_x, max_x = 0, self.circ.full_range
                xs, density_vals = vonmises_kde(values, kappa, min_x, max_x, n_bins)

                self.ax.plot(
                    xs,
                    density_vals,
                    color=self.colours[idx % len(self.colours)],
                    label=label,
                    **kwargs,
                )
        else:
            values = self.circ.data
            if len(values) == 0:
                return

            min_x, max_x = 0, self.circ.full_range
            xs, density_vals = vonmises_kde(values, kappa, min_x, max_x, n_bins)

            self.ax.plot(xs, density_vals, color=DEFAULT_COLOUR, label="all", **kwargs)

    def add_histogram(
        self,
        data: Optional[Union[dict[str, np.ndarray], np.ndarray]] = None,
        bins=36,
        alpha=0.2,
        color: str = "grey",
    ):
        """
        Plot histogram as radial bars on the polar axis.
        If `data` is None, uses circular data in `self.circs`.

        Parameters:
        - data: Optional dict[label] = array or a single array.
        - bins: Number of angular bins (default 36).
        - alpha: Transparency of bars.
        """
        if data is None:
            raise NotImplementedError(
                "Not yet implemented plotting the data in circ. Please supply data. "
            )
            data_to_plot = {label: circ.data for label, circ in self.circs.items()}
        elif isinstance(data, dict):
            data_to_plot = data
        elif isinstance(data, np.ndarray):
            data_to_plot = {"data": data}
        else:
            raise ValueError(
                "Input `data` must be None, a numpy array, or a dict of arrays."
            )

        for idx, (label, values) in enumerate(data_to_plot.items()):
            counts, bin_edges = np.histogram(values, bins=bins, range=(0, 2 * np.pi))

            r_min, r_max = (
                self.ax.get_ylim()
            )  # Set max radius of bars (so they fit on same scale as points)
            max_height = r_max * 0.7  # Keep some space for clarity
            if counts.max() > 0:
                counts = (counts / counts.max()) * max_height

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            width = (2 * np.pi) / bins

            self.ax.bar(
                bin_centers,
                counts,
                width=width,
                bottom=0,
                align="center",
                alpha=alpha,
                color=color,
                edgecolor=color,
                label=f"{label}",
            )

    def add_circular_mean(self, grouped: Optional[bool] = None, **kwargs):
        """
        Plot mean resultant vector(s) as arrows.

        Parameters
        ----------
        grouped : bool, optional
            If None, uses `self.group_by_labels`.
            If True, plots a vector per label.
            If False, plots a single mean vector for all data.
        kwargs : dict
            Additional keyword arguments passed to `ax.arrow`.
            Common kwargs: width, color, alpha, linewidth.
        """
        if grouped is None:
            grouped = self.group_by_labels

        arrow_defaults = dict(width=0.02, head_length=0.0)  # small arrow body
        arrow_defaults.update(kwargs)

        if grouped:
            if self.circ.labels is None:
                raise ValueError(
                    "Cannot group by labels: no labels present in the Circular object."
                )

            unique_labels = np.unique(self.circ.labels)
            for idx, label in enumerate(unique_labels):
                values = self.circ.data[self.circ.labels == label]
                if len(values) == 0:
                    continue

                mean_angle = circular_mean(values)
                r = circular_r(values)

                self.ax.arrow(
                    mean_angle,
                    0,
                    0,
                    r,
                    color=self.colours[idx % len(self.colours)],
                    label=label,
                    **arrow_defaults,
                )
        else:
            values = self.circ.data
            if len(values) == 0:
                return

            mean_angle = circular_mean(values)
            r = circular_r(values)

            self.ax.arrow(mean_angle, 0, 0, r, color=DEFAULT_COLOUR, **arrow_defaults)

    def add_legend(self, location="upper right", **kwargs):
        """
        Add a legend to the plot.

        Parameters
        ----------
        location : str
            Position of the legend (default is 'upper right').
        kwargs : dict
            Additional arguments passed to `ax.legend()`.
        """
        self.ax.legend(loc=location, **kwargs)

    def show(self):
        plt.show()

    def save(self, filename, **kwargs):
        """Save the figure to a file"""
        self.fig.savefig(filename, **kwargs)


# other plotting functions (mainly for ts)
