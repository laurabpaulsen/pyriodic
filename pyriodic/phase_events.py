
import numpy as np
from typing import Union, Optional
from .circular import Circular
import pandas as pd

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
    metadata: Optional[pd.DataFrame] = None,
) -> Union["Circular", tuple["Circular", list[int]]]:
    """
    Create Circular object from a phase angle time series and event markers,
    optionally applying rejection criteria.

    Parameters
    ----------
    phase_ts : np.ndarray
        1D array of phase values (in radians), typically spanning multiple 0–2π cycles.
    events : np.ndarray
        1D array of sample indices at which to extract phase values.
    event_labels : np.ndarray, optional
        Labels corresponding to each event, for grouping in the Circular object. Must be the same length as `events`.
    unit : str
        Unit of the input phase time series and the desired unit for output Circular objects.
        Must be either 'radians' or 'degrees'. Internally, all computations are performed in radians.
    first_samp : int
        Offset to subtract from each event index to align with the phase time series
        (useful if phase_ts is a segment of a longer recording).
    rejection_method : str, optional
        Method to reject events based on surrounding phase dynamics.
        Supported values: 'segment_duration_sd' — excludes events during rising/falling phase segments whose durations deviate more than `rejection_criterion` standard deviations from the mean.
    rejection_criterion : float, optional
        Threshold (in standard deviation units) for identifying outlier segments.
        Only used if `rejection_method='segment_duration_sd'`. Default is 3.
    bad_segments : Optional[np.ndarray]
        For ignoring events that fall within specific segments of the phase time series. Shape should be (n_segments, 2), where each row is a (start, stop) index pair in samples.
    return_rejected : bool
        If True, also return a list of rejected event indices.

    Returns
    -------
    Circular 
        A Circular object containing the phase values at the specified events. 
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

    # Rejection setup
    rising_outlier_samples = set()
    falling_outlier_samples = set()

    if metadata is not None:
        # events length must match metadata length
        if len(metadata) != len(events):
            raise ValueError("Length of metadata must match length of events.")
        
        metadata_rows = []
        
        

    if rejection_method == "segment_duration_sd":
        if rejection_criterion is None:
            rejection_criterion = 3  # default to 3 SD

        rising_segments, falling_segments = get_phase_segments(phase_ts)
        rising_outliers = compute_segment_outliers(rising_segments, threshold=rejection_criterion)
        falling_outliers = compute_segment_outliers(falling_segments, threshold=rejection_criterion)

        rising_outlier_samples = get_outlier_sample_indices(rising_segments, rising_outliers)
        falling_outlier_samples = get_outlier_sample_indices(falling_segments, falling_outliers)

    accepted_phase_val = []
    accepted_labels = []
    for i, (event, label) in enumerate(zip(events, labels)): 
        
        if metadata is not None:
            meta = metadata.iloc[i]
        else:
            meta = None

        idx = event - first_samp
        if not (0 <= idx < len(phase_ts)):
            continue  # skip out-of-bounds events

        phase_val = phase_ts[idx]

        if rejection_method == "segment_duration_sd":
            if idx in rising_outlier_samples or idx in falling_outlier_samples:
                n_rejected += 1
                rejected_indices.append(event)
                continue

        if bad_segments is not None:
            in_bad = any(start <= idx < stop for start, stop in bad_segments)
            if in_bad:
                n_rejected += 1
                rejected_indices.append(event)
                continue

        accepted_phase_val.append(phase_val)
        accepted_labels.append(label)
        if meta is not None:
            metadata_rows.append(meta)

    print(f"Rejected {n_rejected} out of {len(events)} events "
          f"({n_rejected / len(events) * 100:.1f}%)")
    
    updated_metadata = pd.DataFrame(metadata_rows) if metadata is not None else None


    circ_obj = Circular(
        np.rad2deg(accepted_phase_val) if unit == "degrees" else accepted_phase_val,
        unit=unit,
        labels=accepted_labels,
        metadata=updated_metadata,
    )

        

    return circ_obj

