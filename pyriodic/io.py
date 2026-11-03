
import mne
import numpy as np
from typing import Optional
from .preproc import RawSignal


def mne_to_pyriodic(
    mne_raw: mne.io.Raw,
    mne_events: Optional[np.ndarray] = None,
    trigger_mapping: Optional[dict[int, str]] = None,
    channel_name: str = "resp",
    bad_annotation_labels: Optional[list[str]] = None,
):
    """
    Extract a time series from an MNE Raw object and optionally prepare
    event samples and labels for analysis with pyriodic.

    Parameters
    ----------
    mne_raw : mne.io.Raw
        The MNE Raw object containing the signal. To be implemented: Path to the MNE Raw file (e.g., .fif, .edf, .bdf) to be loaded.
    mne_events : np.ndarray, optional
        MNE events array with shape (n_events, 3). The first column
        contains sample indices and the third column contains event IDs.
    trigger_mapping : dict[int, str], optional
        Mapping from event IDs to event labels. Event IDs not present
        in the mapping are labelled "unknown".
    channel_name : str
        Name of the channel from which to extract the time series.
    bad_annotation_labels : list of str, optional
        Annotation descriptions in the `mne_raw` object that should be treated as bad segments. Matching is case-insensitive.
        For example: ["bad", "artifact", "bad segment"].

        If None, no annotations are marked as bad.

    Returns
    -------
    ts : np.ndarray
        One-dimensional time series from `channel_name`.
    samples : np.ndarray or None
        Event sample indices relative to the beginning of `mne_raw`,
        or None if no events were provided.
    labels : np.ndarray or None
        Event labels, or None if no events were provided.
    """

    ts = mne_raw.copy().pick([channel_name]).get_data()[0]
    ts = RawSignal(ts, mne_raw.info["sfreq"])

    samples, labels, bad_segments = None, None, None

    if mne_events is not None:
        if mne_events.ndim != 2 or mne_events.shape[1] < 3:
            raise ValueError(
                "`mne_events` must have shape (n_events, 3)."
            )

        # Convert MNE's absolute sample indices to indices relative
        # to the beginning of this Raw object.
        samples = mne_events[:, 0] - mne_raw.first_samp

        event_ids = mne_events[:, 2]

        if trigger_mapping is None:
            labels = event_ids
        else:
            labels = np.array(
                [trigger_mapping.get(event_id, "unknown") for event_id in event_ids],
                dtype=object,
            )


    # if annotations are provided, mark the segments as bad if they match the provided labels (bad_annotation_labels)
    if bad_annotation_labels is not None and len(mne_raw.annotations) > 0:

        bad_annotation_labels = {
            label.lower() for label in bad_annotation_labels
        }

        bad_segments = []

        for annot in mne_raw.annotations:
            description = annot["description"]

            if description.lower() in bad_annotation_labels:
                start_sample = int(
                    round(annot["onset"] * mne_raw.info["sfreq"])
                )
                end_sample = int(
                    round(
                        (annot["onset"] + annot["duration"])
                        * mne_raw.info["sfreq"]
                    )
                )

                bad_segments.append((start_sample, end_sample))

        
        if bad_segments:
            ts.annotate_bad_segments(
                bad_segments,
                unit="sample",
            )

    if not bad_segments and bad_annotation_labels is not None:
        print(
            "No annotations matched the provided `bad_annotation_labels`."
        )
    return ts, samples, labels

