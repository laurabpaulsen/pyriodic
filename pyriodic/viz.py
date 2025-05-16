import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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

    def _resolve_color(self, idx=None, label=None, override_color=None):
        """
        Determine the color to use for plotting.

        Priority:
        1. override_color passed directly by user
        2. group-based color (via idx or label)
        3. fallback to DEFAULT_COLOUR
        """
        if override_color is not None:
            return override_color

        if label is not None and self.circ.labels is not None:
            # Use label-to-color matching if you want more control
            unique_labels = np.unique(self.circ.labels)
            idx = np.where(unique_labels == label)[0][0]

        if idx is not None and hasattr(self, "colours"):
            return self.colours[idx % len(self.colours)]

        return DEFAULT_COLOUR
    
    def _update_kwargs(self, defaults, kwargs):
        """
        Merges default plotting kwargs with user overrides.
        Extracts and removes 'color' from kwargs to prevent duplication.
        
        Returns:
        - updated kwargs dict
        - color override (if any)
        """
        overwrite_color = kwargs.pop("color", None)
        updated_kwargs = {**defaults, **kwargs}
        return updated_kwargs, overwrite_color

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
                    color=self._resolve_color(idx=idx, label=label, override_color=kwargs.get("color")),
                    label=label,
                    **kwargs,
                )
        else:
            self.ax.scatter(
                self.circ.data,
                [0.5] * len(self.circ.data),
                color=self._resolve_color(override_color=kwargs.get("color")),
                label="Events",
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

            self.ax.plot(
                xs,
                density_vals,
                color=DEFAULT_COLOUR,
                label="Density of events",
                **kwargs,
            )

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

    def add_arrows(
        self,
        angles: np.ndarray,
        lengths: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Plot arrows at specified angles and lengths.

        Parameters
        ----------
        angles : np.ndarray
            1D array of angles (in radians) where arrows should be drawn.

        lengths : np.ndarray, optional
            Length of each arrow. If None, all arrows will have unit length.

        labels : np.ndarray, optional
            Optional label array (same length as angles) for grouping and coloring.

        kwargs : dict
            Additional keyword arguments passed to `ax.arrow`. Common examples: width, color, alpha.
        """

        arrow_defaults, overwrite_color = self._update_kwargs(dict(width=0.02, head_length=0.0), kwargs)

        if lengths is None:
            lengths = np.ones_like(angles)

        if labels is not None:
            unique_labels = np.unique(labels)
            for idx, label in enumerate(unique_labels):
                mask = labels == label
                for angle, r in zip(angles[mask], lengths[mask]):
                    self.ax.arrow(
                        angle,
                        0,
                        0,
                        r,
                        color=self._resolve_color(idx=idx, label=label, override_color=overwrite_color),
                        label=label,
                        **arrow_defaults,
                    )
        else:
            for angle, r in zip(angles, lengths):
                self.ax.arrow(angle, 0, 0, r, color=self._resolve_color(override_color=overwrite_color), **arrow_defaults)

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
        """
        if grouped is None:
            grouped = self.group_by_labels

        if grouped:
            if self.circ.labels is None:
                raise ValueError(
                    "Cannot group by labels: no labels present in the Circular object."
                )

            angles = []
            lengths = []
            labels = []

            unique_labels = np.unique(self.circ.labels)
            for label in unique_labels:
                values = self.circ.data[self.circ.labels == label]
                if len(values) == 0:
                    continue
                angles.append(circular_mean(values))
                lengths.append(circular_r(values))
                labels.append(label)

            self.add_arrows(
                np.array(angles), np.array(lengths), np.array(labels), **kwargs
            )

        else:
            values = self.circ.data
            if len(values) == 0:
                return

            angle = circular_mean(values)
            length = circular_r(values)
            self.add_arrows(np.array([angle]), np.array([length]), **kwargs)

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


# ------- DIAGNOSIC PLOT FOR PREPROCESSING -----------
def plot_phase_diagnostics(
    phase_angles: dict[str, np.ndarray],
    fs: Union[int, float],
    data: Optional[np.ndarray] = None,
    events: Optional[Union[list, np.ndarray]] = None,
    event_labels: Optional[Union[list, np.ndarray]] = None,
    peaks=None,
    troughs=None,
    flat_start_stop=None,
    savepath=None,
    figsize=None,
    interactive: bool = False,
    window_duration: float = 20.0,
    title=None,
):
    """
    Parameters
    ----------
    phase_angles : dict[str, np.ndarray]
        Dictionary of named phase angle signals (e.g., {"Three-point": ..., "Hilbert": ...}).
        Each should be 1D array of same length as `data`.
    fs : float
        Sampling frequency (Hz).
    data : np.ndarray, optional
        Raw or preprocessed time series signal (same length as phase arrays).
    events : list[int] or np.ndarray, optional
        Sample indices of events to mark (e.g., stimulus or response times).
    event_labels : list[str], optional
        Label for each event (same length as `events`). Used for color grouping.
    peaks : list[int] or np.ndarray, optional
        Sample indices of identified peaks.
    troughs : list[int] or np.ndarray, optional
        Sample indices of identified troughs.
    flat_start_stop : list[tuple[int, int]] or list[int], optional
        Flat segments as start–stop index tuples or flat indices directly.
    savepath : str or Path, optional
        If provided, saves static plot to this location (only applies if interactive=False).
    figsize : tuple[int, int], optional
        Size of the figure in inches (only applies if interactive=False).
    interactive : bool
        If True, launches an interactive slider-based window viewer (Matplotlib).
    window_duration : float
        Duration (in seconds) of each visible window in interactive mode.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : list[matplotlib.axes.Axes]
        The axes used in the plot (one per row: signal + phase tracks).
    """
    n_phase_axes = len(phase_angles)
    n_rows = n_phase_axes + (1 if data is not None else 0)
    time = np.arange(len(next(iter(phase_angles.values())))) / fs

    if interactive:
        window_samples = int(window_duration * fs)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        slider_ax = plt.axes([0.1, 0.01, 0.8, 0.03])
        slider = Slider(
            slider_ax,
            "Time (s)",
            0,
            time[-1] - window_duration,
            valinit=0,
            valstep=1 / fs,
        )

        lines = []
        marker_objs = []
        row_idx = 0
        start_idx = 0
        end_idx = start_idx + window_samples
        if title:
            fig.suptitle(title)

        if data is not None:
            ax = axes[row_idx]
            (line,) = ax.plot(
                time[start_idx:end_idx],
                data[start_idx:end_idx],
                color="green",
                label="Signal",
            )
            lines.append((line, data))

            def scatter_pts(indices, color, label):
                return ax.plot([], [], "o", color=color, label=label, markersize=4)[0]

            # --- Markers
            if peaks is not None:
                peak_sc = scatter_pts(peaks, "blue", "Peaks")
                marker_objs.append((peak_sc, np.asarray(peaks)))
            if troughs is not None:
                trough_sc = scatter_pts(troughs, "purple", "Troughs")
                marker_objs.append((trough_sc, np.asarray(troughs)))
            flat_idxs = []
            if flat_start_stop is not None:
                if isinstance(flat_start_stop[0], (tuple, list)):
                    flat_idxs = [
                        i for start, end in flat_start_stop for i in range(start, end)
                    ]
                else:
                    flat_idxs = flat_start_stop
                flat_sc = scatter_pts(flat_idxs, "orange", "Flat")
                marker_objs.append((flat_sc, np.asarray(flat_idxs)))

            ax.legend(loc="upper right")
            row_idx += 1

        # --- Phase angles
        for label, phase in phase_angles.items():
            (line,) = axes[row_idx].plot(
                time[start_idx:end_idx],
                phase[start_idx:end_idx],
                color="grey",
                label=label,
            )
            axes[row_idx].set_ylabel(label)
            lines.append((line, phase))
            row_idx += 1

        # --- Events
        event_lines = []
        if events is not None:
            if event_labels is None:
                event_labels = ["event"] * len(events)
            unique_labels = sorted(set(event_labels))
            cmap = plt.cm.get_cmap("tab10", len(unique_labels))
            label_colors = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

            for ax in axes:
                ev_objs = []
                for ev_sample, ev_label in zip(events, event_labels):
                    line = ax.axvline(
                        x=0,
                        color=label_colors[ev_label],
                        linestyle="--",
                        alpha=0.5,
                        visible=False,
                    )
                    ev_objs.append((line, ev_sample))
                event_lines.append(ev_objs)

        def update(val):
            start_idx = int(val * fs)
            end_idx = start_idx + window_samples
            for ax, (line, y_data) in zip(axes, lines):
                line.set_xdata(time[start_idx:end_idx])
                line.set_ydata(y_data[start_idx:end_idx])
                ax.set_xlim(time[start_idx], time[end_idx - 1])
                ax.relim()
                ax.autoscale_view()

            for marker, indices in marker_objs:
                if indices is not None and len(indices) > 0:
                    mask = (indices >= start_idx) & (indices < end_idx)
                    in_window = indices[mask]
                    marker.set_xdata(time[in_window])
                    marker.set_ydata(data[in_window])

            for ev_objs in event_lines:
                for line, ev_sample in ev_objs:
                    visible = start_idx <= ev_sample < end_idx
                    line.set_xdata(ev_sample / fs)
                    line.set_visible(visible)

            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.subplots_adjust(bottom=0.12)
        plt.show()
        return fig, axes

    else:
        # Static full-range view
        if figsize is None:
            figsize = (12, 2.5 * n_rows)

        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, dpi=300, sharex=True)
        if n_rows == 1:
            axes = [axes]

        ax_idx = 0

        if data is not None:
            axes[ax_idx].plot(
                time, data, label="Signal", color="forestgreen", linewidth=1
            )
            for arr, label, color in zip(
                [peaks, flat_start_stop, troughs],
                ["Peaks", "Flat", "Troughs"],
                ["blue", "orange", "purple"],
            ):
                if arr is not None and len(arr) > 0:
                    if isinstance(arr[0], (tuple, list)):
                        flat_idx = [i for start, end in arr for i in range(start, end)]
                        y_vals = [data[i] for i in flat_idx]
                        axes[ax_idx].scatter(
                            time[flat_idx], y_vals, s=3, label=label, color=color
                        )
                    else:
                        y_vals = [data[i] for i in arr]
                        axes[ax_idx].scatter(
                            time[arr], y_vals, s=3, label=label, color=color
                        )
            axes[ax_idx].legend(loc="upper right")
            ax_idx += 1

        for label, values in phase_angles.items():
            axes[ax_idx].plot(time, values, color="grey", linewidth=1)
            axes[ax_idx].set_ylabel(label)
            ax_idx += 1

        if events is not None:
            if event_labels is None:
                event_labels = ["event"] * len(events)
            unique_labels = list(set(event_labels))
            colors = plt.cm.get_cmap("tab10", len(unique_labels))
            label_color_map = {lbl: colors(i) for i, lbl in enumerate(unique_labels)}
            for ev_idx, ev_sample in enumerate(events):
                lbl = event_labels[ev_idx]
                for ax in axes:
                    ax.axvline(
                        ev_sample / fs,
                        color=label_color_map[lbl],
                        linestyle="--",
                        alpha=0.5,
                    )

        axes[-1].set_xlabel("Time (s)")
        for ax in axes:
            ax.set_xlim((0, time[-1]))

        plt.tight_layout()
        if savepath:
            fig.savefig(savepath)
            plt.close(fig)
        else:
            return fig, axes
