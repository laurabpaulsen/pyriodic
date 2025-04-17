import matplotlib.pyplot as plt
import numpy as np
from .circular import Circular
from math import pi

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


class PyCircPlot:
    def __init__(
        self,
        circs: dict[str, Circular],
        colours=None,
        fig_size=(6, 6),
        dpi=300,
        ax=None,
        ylim=None,
    ):
        self.circs = circs

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
            self.colours = color_cycle[: len(circs)]

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

    def add_points(self):
        for idx, (label, circ) in enumerate(self.circs.items()):
            self.ax.scatter(
                circ.data, [0.5] * len(circ.data), color=self.colours[idx], label=label
            )

    def add_density(self, kappa=20, n_bins=500):
        """
        Add a circular density estimate using Von Mises KDE.

        Parameters:
        - colour: Line color for the density plot.
        - kappa: Concentration parameter of the Von Mises distribution (0 = uniform).
        - n_bins: Number of bins to evaluate density on.
        """
        for idx, (label, circ) in enumerate(self.circs.items()):

            if circ.unit == "radians":
                min_x, max_x = 0, 2 * pi

            elif circ.unit == "degrees":
                min_x, max_x = 0, 360

            else:
                raise ValueError("upss")

            xs, density_vals = vonmises_kde(
                circ.data, kappa, min_x=min_x, max_x=max_x, n_bins=n_bins
            )

            self.ax.plot(
                xs, density_vals, color=self.colours[idx], linewidth=1.5, label=label
            )

        self.ax.legend()

    def add_circular_mean(self):
        # calculate the mean using function from Circular
        pass

    def show(self):
        plt.show()

    def save(self, filename, **kwargs):
        """Save the figure to a file"""
        self.fig.savefig(filename, **kwargs)


# other plotting functions (mainly for ts)
