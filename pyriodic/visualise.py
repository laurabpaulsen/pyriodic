
import matplotlib.pyplot as plt
import numpy as np
from circular import Circular
from math import pi
DEFAULT_COLOUR = "forestgreen"



def vonmises_kde(data, kappa, min_x = -pi, max_x = pi, n_bins=100):
    from scipy.special import i0
    bins = np.linspace(min_x, max_x, n_bins)
    x = np.linspace(min_x, max_x, n_bins)
    # integrate vonmises kernels
    kde = np.exp(kappa*np.cos(x[:, None]-data[None, :])).sum(1)/(2*np.pi*i0(kappa))
    kde /= np.trapz(kde, x=bins)
    return bins, kde

class PyCircPlot:
    def __init__(self, circ: Circular, fig_size=(6, 6), dpi=300, ax=None, ylim = None):
        self.circ = circ

        if ax is None:
            self.fig, self.ax = plt.subplots(
                figsize=fig_size, dpi=dpi, subplot_kw={"projection": "polar"}
            )
        else:
            # Ensure it's a polar axis
            if ax.name != 'polar':
                raise ValueError("Provided axis must be a polar projection")
            self.ax = ax
            self.fig = ax.figure

        self.prepare_ax(ylim = ylim)

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
        zero_location = pi if self.circ.zero == "pi" else self.circ.zero
        self.ax.set_theta_offset(zero_location)

        # Direction of theta
        self.ax.set_theta_direction(-1)

    def add_points(self, colour = DEFAULT_COLOUR):
        self.ax.scatter(self.circ.data, [0.5]*len(self.circ.data), color = colour)
    
    from numpy import pi

    def add_density(self, colour=DEFAULT_COLOUR, kappa=20, n_bins=500):
        """
        Add a circular density estimate using Von Mises KDE.

        Parameters:
        - colour: Line color for the density plot.
        - kappa: Concentration parameter of the Von Mises distribution (like 1/variance).
        - n_bins: Number of bins to evaluate density on.
        """

        if self.circ.unit == "radians":
            tmp_angles = self.circ.data

            if self.circ.zero == 0:
                min_x, max_x = -pi, pi
            elif self.circ.zero == "pi":
                min_x, max_x = 0, 2 * pi
            else:
                raise ValueError("Unsupported value for circ.zero. Expected 0 or 'pi'.")

            xs, density_vals = vonmises_kde(tmp_angles, kappa, min_x=min_x, max_x=max_x, n_bins=n_bins)

            self.ax.plot(xs, density_vals, color=colour, linewidth=1.5)

        else:
            print("upss")

    def add_circular_mean(self):
        # calculate the mean using function from Circular
        pass

    def show(self):
        plt.show()

    def save(self, filename, **kwargs):
        """Save the figure to a file"""
        self.fig.savefig(filename, **kwargs)



# other plotting functions (mainly for ts)
