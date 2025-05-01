from collections import defaultdict
import numpy as np
from typing import Union, Optional
import matplotlib.pyplot as plt

from .circular import Circular

class PhaseEvents:
    def __init__(self, phase_dict: dict):
        """
        Container for condition-sorted Circular phase data.

        Parameters
        ----------
        phase_dict : dict
            Dictionary mapping condition labels or codes to Circular objects.
        """
        self.phase_dict = phase_dict

    def mean(self):
        return {label: circ.mean() for label, circ in self.phase_dict.items()}

    def r(self):
        return {label: circ.r() for label, circ in self.phase_dict.items()}

    def plot(self):
        n = len(self.phase_dict)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, subplot_kw={'projection': 'polar'}, figsize=(4 * cols, 4 * rows), dpi = 300)
        axes = np.array(axes).reshape(-1)  # flatten in case of 2D array

        for ax, (label, circ) in zip(axes, self.phase_dict.items()):
            circ.plot(label=label, ax = ax)
            ax.set_title(f"{label}")

        # Hide unused axes
        for ax in axes[len(self.phase_dict):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()


    def __getitem__(self, key):
        return self.phase_dict[key]

    def __repr__(self):
        return f"<PhaseEvents: conditions={list(self.phase_dict.keys())}>"



def create_phase_events(
    phase_ts: np.ndarray,
    events: np.ndarray,
    event_ids: Optional[Union[np.ndarray, dict]] = None,
    unit: str = "radians",
    first_samp: int = 0
) -> Union[Circular, PhaseEvents]:
    """
    Create Circular object(s) from a phase angle time series and event markers.

    Parameters
    ----------
    phase_ts : np.ndarray
        1D array of phase values in radians.
    events : np.ndarray
        Event sample indices.
    event_ids : np.ndarray or dict, optional
        Event condition codes or label mapping.
        - If None: returns a single Circular object.
        - If array: used to group events by value.
        - If dict: keys are condition labels, values are event codes.
    unit : str
        Unit to assign to Circular objects ('radians', 'degrees', etc.).
    first_samp : int
        Offset for non-zero-indexed phase time series.

    Returns
    -------
    Circular or PhaseEvents
        Single Circular object if event_ids is None, otherwise PhaseEvents container.
    """
    events = np.asarray(events)

    if event_ids is None:
        phase_data = phase_ts[events - first_samp]
        return Circular(phase_data, unit=unit)

    if isinstance(event_ids, dict):
        code_to_label = {v: k for k, v in event_ids.items()}
        labels = np.array([code_to_label.get(ev, "unknown") for ev in events])
    else:
        labels = np.asarray(event_ids)

    if len(labels) != len(events):
        raise ValueError("event_ids and events must be the same length if event_ids is an array.")

    grouped_phases = defaultdict(list)
    for phase_val, label in zip(phase_ts[events - first_samp], labels):
        grouped_phases[label].append(phase_val)

    circular_dict = {
        label: Circular(angles, unit=unit) for label, angles in grouped_phases.items()
    }

    return PhaseEvents(circular_dict)


