import numpy as np
from typing import Union, Optional, Callable
from scipy import interpolate, signal
from scipy.signal import hilbert
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class RawSignal:
    def __init__(self, data, fs, info=None):
        self.ts = np.asarray(data).copy()

        self.fs = fs
        self.info = info or {}
        # self._phase = None
        # self._peaks = None
        # self._troughs = None
        self._history = []

    def copy(self):
        return deepcopy(self)

    def remove_outliers(self, threshold=2.5, linear_interpolation=True):
        """
        Removes outliers .... threshold + linear interpolation
        """
        z_scores = np.abs((self.ts - np.nanmean(self.ts)) / np.nanstd(self.ts))
        self.ts[z_scores > threshold] = np.nan

        # linear interpolation of NaNs
        if linear_interpolation:
            self.interpolate_missing()

        self._history.append(
            f"remove_outliers({threshold}), {sum(z_scores > threshold)} outliers found"
        )

    def interpolate_missing(self):
        """ """
        nans = np.isnan(self.ts)
        x_vals = np.where(~nans)[0]
        y_vals = self.ts[~nans]
        interpolated_values = interpolate.interp1d(
            x_vals, y_vals, kind="linear", fill_value="extrapolate"
        )
        self.ts[nans] = interpolated_values(np.where(nans)[0])

        self._history.append(f"interpolate_missing(), {np.sum(nans)} NaNs interpolated")

    def zscore(self):
        """
        Normalizes the signal in-place to mean 0 and unit variance.
        """
        self.ts = (self.ts - np.nanmean(self.ts)) / np.nanstd(self.ts)
        self._history.append("normalize()")

    def filter_bandpass(self, low, high):
        """ """
        sos = signal.butter(N=4, Wn=[low, high], btype="band", fs=self.fs, output="sos")
        self.ts = signal.sosfiltfilt(sos, self.ts)
        self._history.append(f"bandpass({low}-{high})")

    def phase_hilbert(self):
        """
        Extract instantaneous phase angle using the Hilbert Transform.

        Parameters:
            ts (np.ndarray): 1D time series

        Returns:
            np.ndarray: phase angles in radians (0 to 2π)
        """

        # Hilbert transform
        analytic_signal = hilbert(self.ts)
        phase_angles = np.angle(analytic_signal)  # returns in range [-π, π]

        return phase_angles % (2 * np.pi)  # return from 0 to 2pi instead

    def phase_linear(self, peak_finder=None, distance=100, prominence=0.01):
        """
        Extract phase using linear interpolation between peaks and troughs.

        Args:
            peak_finder (callable): Optional custom function(ts, **kwargs) -> np.ndarray of peak indices.
            distance (int): Minimum distance between peaks.
            prominence (float): Prominence threshold for peak detection.

        Returns:
            phase
            peaks
            troughs
        """
        if peak_finder is None:
            # Default logic
            def peak_finder(ts, distance=distance, prominence=prominence):
                peaks, _ = signal.find_peaks(
                    ts, distance=distance, prominence=prominence
                )
                return peaks

        peaks = peak_finder(self.ts, distance=distance, prominence=prominence)

        # Troughs between peaks
        troughs = []
        for p1, p2 in zip(peaks[:-1], peaks[1:]):
            segment = self.ts[p1:p2]
            trough = p1 + np.argmin(segment)
            troughs.append(trough)

        # Assign phase
        phase = np.full_like(self.ts, np.nan, dtype=np.float32)
        for p1, p2, t in zip(peaks[:-1], peaks[1:], troughs):
            phase[p1:t] = np.linspace(0, np.pi, t - p1)
            phase[t:p2] = np.linspace(np.pi, 2 * np.pi, p2 - t)
            phase[p1] = 0
            phase[t] = np.pi
            phase[p2] = 2 * np.pi

        # self._phase = phase
        # self._peaks = peaks
        # self._troughs = np.array(troughs)

        return (phase, peaks, troughs)

    def phase_threepoint(
        self,
        peak_finder: Optional[Callable[..., np.ndarray]] = None,
        distance: int = 100,
        prominence: Union[float, int] = 0.01,
        percentile: Union[float, int] = 50,
        descent_window: int = 5,
    ):
        """
        Extract phase using a three-point method:
        Peak → descending slope → flat region → ascending slope → next peak.

        Args:
            peak_finder (callable): Optional function(ts, **kwargs) → np.ndarray of peak indices.
            distance (int): Min distance between peaks (used if no custom peak_finder).
            prominence (float): Prominence threshold for peak detection.
            percentile (float): Percentile of absolute gradient below which region is considered 'flat'.
            descent_window (int): Number of samples used to confirm descent/ascent before/after flat region.

        Returns:
            phase (np.ndarray): Phase array in radians (0 to 2π)
            peaks (np.ndarray): Indices of detected peaks
            troughs (list of tuples): List of (trough_start, trough_end) for flat segments
        """
        if peak_finder is None:

            def peak_finder(ts, distance=distance, prominence=prominence):
                peaks, _ = signal.find_peaks(
                    ts, distance=distance, prominence=prominence
                )
                return peaks

        peaks = peak_finder(self.ts, distance=distance, prominence=prominence)
        if len(peaks) < 2:
            raise ValueError("Need at least two peaks to compute phase.")

        gradient = np.gradient(self.ts)
        troughs = []

        for i in range(len(peaks) - 1):
            start, end = peaks[i], peaks[i + 1]
            segment_grad = gradient[start:end]
            segment_abs_grad = np.abs(segment_grad)
            grad_threshold = np.percentile(segment_abs_grad, percentile)

            flat_indices = np.where(segment_abs_grad < grad_threshold)[0]

            # Step 1: Start of flat region (after descent)
            trough_start = None
            for idx in flat_indices:
                if idx < descent_window:
                    continue
                if np.all(segment_grad[idx - descent_window : idx] < 0):
                    trough_start = start + idx
                    break

            if trough_start is None:
                mid = (start + end) // 2
                troughs.append((mid, mid))
                continue

            # Step 2: End of flat region (before ascent)
            flat_following = flat_indices[flat_indices > (trough_start - start)]
            trough_end = trough_start  # fallback

            for idx in reversed(flat_following):
                if idx + descent_window >= len(segment_grad):
                    continue
                if np.all(segment_grad[idx : idx + descent_window] > 0):
                    trough_end = start + idx
                    break

            troughs.append((trough_start, trough_end))

        phase = np.full(len(self.ts), np.nan, dtype=np.float32)

        for (p1, p2), (t_start, t_end) in zip(zip(peaks[:-1], peaks[1:]), troughs):
            if t_start > p1:
                phase[p1:t_start] = np.linspace(0, np.pi, t_start - p1)
            if t_end > t_start:
                phase[t_start:t_end] = np.pi
            if p2 > t_end:
                phase[t_end:p2] = np.linspace(np.pi, 2 * np.pi, p2 - t_end)

            # Force exact values
            phase[p1] = 0
            phase[t_start] = np.pi
            phase[t_end] = np.pi
            phase[p2] = 2 * np.pi

        return phase, peaks, troughs

    @property
    def history(self):
        if self._history == []:
            raise ValueError("No changes has been made to the data")

        else:
            return self._history

    def __repr__(self):
        return (
            f"<RawSignal | fs={self.fs} Hz, len={len(self.ts)}, "
            f"steps={len(self._history)}>"
        )
