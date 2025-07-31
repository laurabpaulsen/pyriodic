import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from .circular import Circular
from .density import vonmises_kde
from typing import Union, Optional
from .desc import circular_mean, circular_r


DEFAULT_COLOUR = "forestgreen"


class CircPlot:
    def __init__(
        self,
        circ: Circular,
        title: Optional[str] = None,
        group_by_labels: bool = True,
        colours=None,
        fig_size=(6, 6),
        dpi=300,
        ax=None,
        radius_lim=None,
        angles: Optional[list] = None,
        labels: Optional[list] = None
    ):
        """
        Initialise a CircPlot object for polar visualizations of circular data.

        Parameters
        ----------
        circ : Circular
            A Circular object containing circular data and optional labels.
        group_by_labels : bool
            Whether to group plots by `circ.labels` if they exist.
        colours : str or list, optional
            Color or list of colors to use for plotting. If None, uses default matplotlib cycle.
        fig_size : tuple, default (6, 6)
            Size of the figure in inches.
        dpi : int, default 300
            Dots per inch for figure resolution.
        ax : matplotlib.axes._subplots.PolarAxesSubplot, optional
            Polar axis to use. If None, a new figure and axis are created.
        radius_lim : tuple of float, optional
            Radial axis limits as (min, max).
        angles : list of float, optional
            List of theta gridline positions in degrees.
        labels : list of str, optional
            Labels corresponding to `angles`. Should be same length.
        """
        self.circ = circ
        self.group_by_labels = group_by_labels

        if ax is None:
            self.fig, self.ax = plt.subplots(
                figsize=fig_size, dpi=dpi, subplot_kw={"projection": "polar"}
            )
        else:
            # Ensure it's a polar axis
            if ax.name != "polar":
                raise ValueError("Provided axis must be a polar projection. " \
                "If you have created the ax using plt.subplot, try adding project = 'polar'. " \
                "If you have created the ax using plt.subplots try adding subplot_kw={'projection': 'polar}")
            self.ax = ax
            self.fig = ax.figure

        self.prepare_ax(radius_lim=radius_lim, angles=angles, labels=labels, title=title)

        if colours:
            # Allow either a single colour string or an iterable of colours
            self.colours = [colours] if isinstance(colours, str) else colours
        else:
            # Automatically pick colors from matplotlib's default cycle
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            color_cycle = prop_cycle.by_key()["color"]
            if group_by_labels:
                if self.circ.labels is None:
                    raise ValueError(
                        "Can only group by labels if labels are present in the circular object. Either add labels to the circular object, or set `group_by_labels=False` when initialising the CircPlot."
                    )
                self.colours = color_cycle[: len(np.unique(self.circ.labels))]

    def prepare_ax(self, radius_lim=None, angles=None, labels=None, title=None):
        """
        Prepare and customize a polar plot axis (`self.ax`) with standard aesthetic settings.

        Parameters:
        -----------
        radius_lim : tuple or None
            Optional tuple (min, max) for setting radial (r-axis) limits.
        angles : list of float or None
            Angles (in degrees) where labels should be placed on the theta axis.
            Default is [0, 90, 180, 270, 360].
        labels : list of str or None
            Labels corresponding to `angles`. Default is ["0/2π", "", "π", "", ""].

        """
        if angles is None:
            angles = [0, 90, 180, 270, 360]
        if labels is None:
            labels = ["0/2π", "", "π", "", ""]

        # Remove radial ticks and gridlines
        self.ax.set_yticks([])
        self.ax.set_yticklabels([])
        self.ax.yaxis.grid(False)

        # Optionally set radial axis limits
        if radius_lim:
            self.ax.set_ylim(*radius_lim)

        if title:
            self.ax.set_title(title)

        # Set custom theta gridlines and labels
        self.ax.set_thetagrids(angles, labels=labels)

        # Set theta direction to clockwise
        self.ax.set_theta_direction(-1)

    def _pop_kwarg(self, kwargs, key, default=None):
        return kwargs.pop(key, default)

    def _extract_kwargs(self, kwargs, *keys):
        return {key: kwargs.pop(key, None) for key in keys}
    
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
            unique_labels = np.unique(self.circ.labels)
            if label in unique_labels:
                idx = np.where(unique_labels == label)[0][0]
            else:
                print(f"Label '{label}' not found in data labels. Falling back to default color.")
                return DEFAULT_COLOUR

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

        label = self._pop_kwarg(kwargs, "label", "Events")
        color = self._pop_kwarg(kwargs, "color", None)

        if grouped:
            if self.circ.labels is None:
                raise ValueError(
                    "Cannot group by labels: no labels present in the Circular object."
                )

            unique_labels = np.unique(self.circ.labels)
            for idx, group_label in enumerate(unique_labels):
                values = self.circ.data[self.circ.labels == group_label]
                self.ax.scatter(
                    values,
                    [0.5] * len(values),
                    color=self._resolve_color(
                        idx=idx, label=group_label, override_color=color
                    ),
                    label=group_label,
                    **kwargs,
                )
        else:
            self.ax.scatter(
                self.circ.data,
                [0.5] * len(self.circ.data),
                color=self._resolve_color(override_color=color),
                label=label,
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

        label = self._pop_kwarg(kwargs, "label", "Density")
        color = self._pop_kwarg(kwargs, "color", None)

        if grouped:
            if self.circ.labels is None:
                raise ValueError(
                    "Cannot group by labels: no labels present in the Circular object."
                )

            unique_labels = np.unique(self.circ.labels)
            for idx, group_label in enumerate(unique_labels):
                values = self.circ.data[self.circ.labels == group_label]
                if len(values) == 0:
                    continue

                min_x, max_x = min_x, max_x = 0, self.circ.full_range
                xs, density_vals = vonmises_kde(values, kappa, min_x, max_x, n_bins)

                self.ax.plot(
                    xs,
                    density_vals,
                    color=self.colours[idx % len(self.colours)],
                    label=group_label,
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
                color=self._resolve_color(override_color=color),
                label=label,
                **kwargs,
            )

    def add_histogram(
        self,
        data: Optional[np.ndarray]= None,
        label: Optional[str] = None,
        bins=36,
        alpha=0.2,
        color: str = "grey",
    ):
        """
        Plot histogram as radial bars on the polar axis.
        If `data` is None, uses circular data in `self.circs`.

        Parameters:
        - data: np.array
        - bins: Number of angular bins (default 36).
        - alpha: Transparency of bars.
        """
        if data is None:
            raise NotImplementedError(
                "Not yet implemented plotting the data in circ. Please supply data. "
            )
        
        counts, bin_edges = np.histogram(data, bins=bins, range=(0, 2 * np.pi))

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
            label=label
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

        arrow_kwargs = dict(width=0.02, head_length=0.0)
        color = self._pop_kwarg(kwargs, "color", None)
        arrow_kwargs.update(kwargs)


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
                        color=self._resolve_color(
                            idx=idx, label=label, override_color=color
                        ),
                        label=label,
                        **arrow_kwargs,
                    )
        else:
            for angle, r in zip(angles, lengths):
                self.ax.arrow(
                    angle,
                    0,
                    0,
                    r,
                    color=self._resolve_color(override_color=color),
                    **arrow_kwargs,
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

    def add_legend(self, **kwargs):
        """
        Add a legend to the plot.

        Parameters
        ----------

        kwargs : dict
            Additional arguments passed to `ax.legend()`.
        """
        self.ax.legend(**kwargs)

    def ticks_to_degrees(self, n_ticks=8):
        """
        Convert the theta ticks from radians to degrees.
        """
        
        # set ticks to degrees (8 ticks)
        ticks = np.linspace(0, 360, n_ticks, endpoint=False)
        self.ax.set_xticks(np.deg2rad(ticks))
        self.ax.set_xticklabels([f"{int(t)}°" for t in ticks])

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
    window_duration: float = 20.0,
    title=None,
    start = 0,
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
    window_duration : float
        Duration (in seconds) of each visible window in interactive mode.
    start : float
        Start time (in seconds) for the initial view of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : list[matplotlib.axes.Axes]
        The axes used in the plot (one per row: signal + phase tracks).
    """
    if type(phase_angles) is not dict:
        phase_angles = {"Phase": phase_angles}

    n_phase_axes = len(phase_angles)
    n_rows = n_phase_axes + (1 if data is not None else 0)
    time = np.arange(len(next(iter(phase_angles.values())))) / fs

    start = min(start, time[-1] - window_duration)

    window_samples = int(window_duration * fs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    slider_ax = plt.axes((0.1, 0.01, 0.8, 0.03))
    slider = Slider(
            slider_ax,
            "Time (s)",
            0,
            time[-1] - window_duration,
            valinit=int(start),
            valstep=1 / fs,
        )
    
    zoom_ax = plt.axes((0.92, 0.1, 0.02, 0.7))  # (left, bottom, width, height)
    zoom_slider = Slider(
        ax=zoom_ax,
        label="Zoom\n[s]",
        valmin=1.0,
        valmax=min(60.0, time[-1]),  # up to 60s or full length
        valinit=window_duration,
        orientation="vertical",
    )

    lines = []
    marker_objs = []
    row_idx = 0
    start_idx = int(start * fs)
    print("start_idx", start_idx)
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

        def scatter_pts(ax, indices, color, label, start_idx, end_idx):
            indices = np.asarray(indices)
            mask = (indices >= start_idx) & (indices < end_idx)
            in_window = indices[mask]
            sc = ax.plot(
                time[in_window],
                data[in_window],
                "o",
                color=color,
                label=label,
                markersize=4,
            )[0]
            return sc, indices


        # --- Markers
        if peaks is not None:
            peak_sc, peak_indices = scatter_pts(ax, peaks, "blue", "Peaks", start_idx, end_idx)
            marker_objs.append((peak_sc, peak_indices))

        if troughs is not None:
            trough_sc, trough_indices = scatter_pts(ax, troughs, "purple", "Troughs", start_idx, end_idx)
            marker_objs.append((trough_sc, trough_indices))

        if flat_start_stop is not None:
            if isinstance(flat_start_stop[0], (tuple, list)):
                flat_idxs = [i for start, end in flat_start_stop for i in range(start, end)]
            else:
                flat_idxs = flat_start_stop
            flat_sc, flat_indices = scatter_pts(ax, flat_idxs, "orange", "Flat", start_idx, end_idx)
            marker_objs.append((flat_sc, flat_indices))

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
        window_duration_new = zoom_slider.val
        window_samples = int(window_duration_new * fs)

        start_idx = int(slider.val * fs)
        end_idx = start_idx + window_samples
        if end_idx > len(time):
            end_idx = len(time)
            start_idx = max(0, end_idx - window_samples)

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
    zoom_slider.on_changed(update)

    plt.subplots_adjust(bottom=0.12, right=0.9)
    plt.show(block = True)
    return fig, axes

  