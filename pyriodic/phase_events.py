from collections import defaultdict
import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt
from pathlib import Path
from .circular import Circular


class PhaseEvents:
    def __init__(self, phase_dict: dict):
        """
        Container for condition-sorted Circular phase data.

        Parameters
        ----------
        phase_dict
            Dictionary mapping condition labels or codes to Circular objects.
        """
        self.phase_dict = phase_dict

    def mean(self):
        return {label: circ.mean() for label, circ in self.phase_dict.items()}

    def r(self):
        return {label: circ.r() for label, circ in self.phase_dict.items()}

    def plot(
        self,
        histogram: Optional[np.ndarray] = None,
        savepath: Optional[Union[Path, str]] = None,
    ):
        n = len(self.phase_dict)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(
            rows,
            cols,
            subplot_kw={"projection": "polar"},
            figsize=(4 * cols, 4 * rows),
            dpi=300,
        )
        axes = np.array(axes).reshape(-1)  # flatten in case of 2D array

        for ax, (label, circ) in zip(axes, self.phase_dict.items()):
            circ.plot(ax=ax, histogram=histogram)
            ax.set_title(f"{label}")

        # Hide unused axes
        for ax in axes[len(self.phase_dict) :]:
            ax.set_visible(False)

        plt.tight_layout()

        if savepath:
            plt.savefig(savepath)
            plt.close()
        else:
            plt.show()

    def to_circular(self, include: Optional[Union[str, list[str]]] = None):
        """
        Convert selected events from the phase dictionary to a Circular object.

        Parameters
        ----------
        include : str, list of str, or None
            - If a string, includes all keys in the phase_dict that contain this substring.
            - If a list of strings, includes matching keys directly.
            - If None, includes all keys in phase_dict.

        Returns
        -------
        Circular
            A Circular object constructed from the selected phase values.
        """
        if include is None:
            include = list(self.phase_dict.keys())
        elif isinstance(include, str):
            include = [k for k in self.phase_dict if include in k]
            if not include:
                raise KeyError(f"No matching labels containing '{include}'")

        matched = [(label, self.phase_dict[label]) for label in include]
        labels, circulars = zip(*matched)
        return Circular.from_multiple(circulars, labels=labels)

    def __getitem__(self, key):
        return self.phase_dict[key]

    def __repr__(self):
        return f"<PhaseEvents: conditions={list(self.phase_dict.keys())}>"


def get_outlier_sample_indices(segments, outlier_indices):
    """Convert outlier segments to a flat set of sample indices for fast lookup."""
    outlier_samples = set()
    for i in outlier_indices:
        start, stop = segments[i]
        outlier_samples.update(range(start, stop))
    return outlier_samples


def compute_segment_outliers(segments, threshold=3, sampling_rate=None):
    """
    Identify segment durations that deviate > threshold * SD from the mean.
    Optionally normalize by sampling_rate.
    """
    durations = np.array([stop - start for start, stop in segments])
    if sampling_rate:
        durations = durations / sampling_rate  # convert to seconds

    mean_dur = durations.mean()
    std_dur = durations.std()

    outlier_indices = [
        i
        for i, dur in enumerate(durations)
        if np.abs(dur - mean_dur) > threshold * std_dur
    ]
    return outlier_indices


def get_phase_segments(phase_ts: np.ndarray):
    """
    Identify rising (0→π) and falling (π→2π) phase segments in a wrapped phase signal.

    Parameters
    ----------
    phase_ts : np.ndarray
        1D array of phase values (in radians), assumed to evolve continuously over time.

    Returns
    -------
    rising_segments : list of (start_idx, stop_idx)
        List of index tuples for rising phase segments (0 to π).
    falling_segments : list of (start_idx, stop_idx)
        List of index tuples for falling phase segments (π to 2π).
    """
    wrapped = np.mod(phase_ts, 2 * np.pi)

    # Detect where phase drops (i.e., 2π → 0 transition): start of new cycle
    phase_diff = np.diff(wrapped)
    cycle_starts = np.where(phase_diff < -np.pi)[0] + 1

    rising_segments = []
    falling_segments = []

    for i in range(len(cycle_starts) - 1):
        start = cycle_starts[i]
        end = cycle_starts[i + 1]

        # Within each cycle, find index closest to π (midpoint)
        cycle_phases = wrapped[start:end]
        mid_rel_idx = np.argmin(np.abs(cycle_phases - np.pi))
        mid = start + mid_rel_idx

        rising_segments.append((start, mid))
        falling_segments.append((mid, end))

    return rising_segments, falling_segments


def create_phase_events(
    phase_ts: np.ndarray,
    events: np.ndarray,
    event_labels: Optional[np.ndarray] = None,
    unit: str = "radians",
    first_samp: int = 0,
    rejection_method: Optional[str] = None,
    rejection_criterion: Optional[float] = None,
    bad_segments: Optional[np.ndarray] = None,
    return_rejected: bool = False,
) -> Union["PhaseEvents", tuple["PhaseEvents", list[int]]]:
    """
    Create Circular object(s) from a phase angle time series and event markers.
    Optionally apply cycle-based rejection criteria to exclude events occurring during atypical phase dynamics.

    Parameters
    ----------
    phase_ts : np.ndarray
        1D array of phase values (in radians), typically spanning multiple 0–2π cycles.
    events : np.ndarray
        1D array of sample indices at which to extract phase values.
    event_labels : np.ndarray, optional
        Labels for grouping events. If None, returns a single Circular object;
        otherwise, returns a PhaseEvents container grouped by unique labels.
    unit : str
        Unit of the input phase time series and the desired unit for output Circular objects.
        Must be either 'radians' or 'degrees'. Internally, all computations are performed in radians.

    first_samp : int
        Offset to subtract from each event index to align with the phase time series
        (useful if phase_ts is a segment of a longer recording).
    rejection_method : str, optional
        Method to reject events based on surrounding phase dynamics.
        Currently supported:
            - 'segment_duration_sd' : excludes events occurring during rising or falling phase
              segments whose durations deviate more than `rejection_criterion` standard deviations
              from the mean across cycles.
    rejection_criterion : float, optional
        Threshold (in standard deviation units) for identifying outlier segments.
        Only used if `rejection_method='segment_duration_sd'`. Default is 3.
    bad_segments : Optional[np.ndarray]
        For ignoring events that fall within specific segments of the phase time series. Shape should be (n_segments, 2), where each row is a (start, stop) index pair in samples.
    return_rejected : bool
        If True, also return a list of rejected event indices.

    Returns
    -------
    Circular or PhaseEvents
        If event_labels is None, returns a single Circular object.
        Otherwise, returns a PhaseEvents object containing condition-grouped Circular data.
    (optional) list[int]
        If `return_rejected` is True, returns a second value: the list of event indices that were rejected.
    """

    if unit == "degrees":
        phase_ts = np.deg2rad(phase_ts)

    elif unit != "radians":
        raise ValueError("unit must be either 'radians' or 'degrees'")

    events = np.asarray(events)

    if event_labels is not None:
        labels = np.asarray(event_labels)
        if len(labels) != len(events):
            raise ValueError("event_labels and events must be the same length.")
    else:
        labels = np.array(["no label"] * len(events))

    n_rejected = 0
    rejected_indices = []

    # Handle rejection logic
    rising_outlier_samples = set()
    falling_outlier_samples = set()

    if rejection_method == "segment_duration_sd":
        if rejection_criterion is None:
            rejection_criterion = 3  # default to 3 SD

        rising_segments, falling_segments = get_phase_segments(phase_ts)
        rising_outliers = compute_segment_outliers(
            rising_segments, threshold=rejection_criterion
        )
        falling_outliers = compute_segment_outliers(
            falling_segments, threshold=rejection_criterion
        )

        rising_outlier_samples = get_outlier_sample_indices(
            rising_segments, rising_outliers
        )
        falling_outlier_samples = get_outlier_sample_indices(
            falling_segments, falling_outliers
        )

    grouped_phases = defaultdict(list)

    for event, label in zip(events, labels):
        idx = event - first_samp
        phase_val = phase_ts[idx]

        if rejection_method == "segment_duration_sd":
            if idx in rising_outlier_samples or idx in falling_outlier_samples:
                n_rejected += 1
                rejected_indices.append(event)
                continue
        
        if bad_segments is not None:
            # check if the event falls within any bad segment
            for start, stop in bad_segments:
                if start <= idx < stop:
                    n_rejected += 1
                    rejected_indices.append(event)
                    continue

        grouped_phases[label].append(phase_val)

    print(
        f"Rejected {n_rejected} out of {len(events)} events ({n_rejected / len(events) * 100:.1f}%)"
    )

    # If all labels are the same ("no label"), return a single Circular object
    unique_labels = set(labels)
    if len(unique_labels) == 1:
        all_angles = grouped_phases[labels[0]]
        circ_obj = Circular(
            np.rad2deg(all_angles) if unit == "degrees" else all_angles, unit=unit
        )
        return (circ_obj, rejected_indices) if return_rejected else circ_obj
    else:
        circular_dict = {
            label: Circular(
                np.rad2deg(angles) if unit == "degrees" else angles, unit=unit
            )
            for label, angles in grouped_phases.items()
        }

        result = PhaseEvents(circular_dict)
        return (result, rejected_indices) if return_rejected else result
