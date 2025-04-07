
import matplotlib.pyplot as plt
import numpy as np
from circular import Circular
from scipy.stats import gaussian_kde

DEFAULT_COLOUR = "forestgreen"

class PyCircPlot:
    def __init__(self, circ: Circular, fig_size=(6, 6), dpi=300, ax=None, zero_pos = "W", ylim = None):
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

        self.prepare_ax(zero_pos = zero_pos.upper(), ylim = ylim)

    def prepare_ax(self, zero_pos, ylim):
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


        # Set theta zero location
        self.ax.set_theta_zero_location(zero_pos)

        # Direction of theta
        self.ax.set_theta_direction(-1)

    def add_points(self, colour = DEFAULT_COLOUR):
        self.ax.scatter(self.circ.data, [0.5]*len(self.circ.data), color = colour)

    def add_density(self, colour = DEFAULT_COLOUR, bw_method = 0.05):
        """
        
        bw_method:
            The method used to calculate the estimator bandwidth. 
            This can be "scott", "silverman", a scalar constant or a callable. If a scalar, this will be used directly as kde.factor. If a callable, it should take a gaussian_kde instance as only parameter and return a scalar. If None (default), "scott" is used.
        """

        if self.circ.unit == "degrees": #NOTE: from -pi to pi. Maybe find a way to include this information in the circular object??
            tmp_angles = self.circ.data
            
            extended_angles = np.concatenate([tmp_angles + 2 * np.pi, tmp_angles, tmp_angles - 2 * np.pi])
            print(extended_angles)

            # Density plot using Gaussian KDE
            density = gaussian_kde(extended_angles, bw_method=bw_method)
            xs = np.linspace(-2 * np.pi, 2 * np.pi, 2000)  # Extend range # something seems offfffff # data given in degrees not between -pi and pi we need to modulate for plot
            density_vals = density(xs)

            # Mask to plot only [-pi, pi]
            mask = (xs >= -np.pi) & (xs <= np.pi)
            self.ax.plot(xs[mask], density_vals[mask], color=colour, linewidth=1.5)

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
