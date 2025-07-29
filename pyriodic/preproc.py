import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Callable, Literal
from scipy import interpolate, signal
from scipy.signal import hilbert
import logging
from copy import deepcopy


logger = logging.getLogger(__name__)


class RawSignal:
    def __init__(self, data, fs, info=None, bad_segments:Union[None, np.ndarray]=None):
        self.ts = np.asarray(data).copy()
        self.fs = fs
        self.info = info or {}
        self.bad_segments = bad_segments
        self._history : list[str] = []

    def copy(self):
        return deepcopy(self)

    def plot(self, ax=None, start=0, duration=20):
        """
        Plots the time series data.

        Args:
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            start: Start time in seconds.
            duration: Duration in seconds to plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        times = np.arange(len(self.ts)) / self.fs


        start_sample = int(start * self.fs)
        end_sample = int((start + duration) * self.fs)

        tmp_ts = self.ts.copy()[start_sample:end_sample]

        ax.plot(times[start_sample:end_sample], tmp_ts)
        ax.set_xlim([start, start + duration])
        #ax.set_ylim([np.min(tmp_ts), np.max(tmp_ts)])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        
        if self.bad_segments is not None:
            for segment in self.bad_segments:
                ax.axvspan(segment[0], segment[1], color='red', alpha=0.5, label='Bad Segment')

        return ax

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
        Normalises the signal in-place to mean 0 and unit variance.
        """
        self.ts = (self.ts - np.nanmean(self.ts)) / np.nanstd(self.ts)
        self._history.append("zscore()")

    def filter_bandpass(self, low = 0.1, high = 1.0):
        """ """
        sos = signal.butter(N=4, Wn=[low, high], btype="band", fs=self.fs, output="sos")
        self.ts = signal.sosfiltfilt(sos, self.ts)
        self._history.append(f"bandpass({low} Hz - {high} Hz)")

    def resample(self, sfreq=Union[int, float]):
        """

        Args:
            sfreq: New sampling rate
        """
        raise NotImplementedError
        # resample

        # update sfreq attribute
        # add attribute that shows that data has been resampled
        # also resample annotations if any
        pass

    def smoothing(self, window_size: float = 50.0):
        """
        Smooths the timeseries by calculating the moving average

        Args:
            window_size (float): Size of the moving window in milliseconds. Default is 50 milliseconds.
        """
        
        # convert window size from milliseconds to samples
        sample_window_size_tmp = window_size * self.fs / 1000
        sample_window_size = int(window_size * self.fs / 1000)

        if sample_window_size_tmp % 1 != 0:
            # print the warning
            raise Warning(
                f"As the window size is not divisible by the sampling frequency, "
                f"the window size will be rounded to the nearest even number: {int(sample_window_size_tmp)} samples, corresponding to {sample_window_size * 1000/self.fs} ms."
            )

        if sample_window_size < 1:
            raise ValueError(
                f"Window size must be at least 1 sample. Increase the window size to a least 1 sample, corresponding to {1000/self.fs} ms. given the sampling frequency of {self.fs} Hz."
                )
                # enforce odd window size so we can center it
        
        if sample_window_size % 2 == 0:
            sample_window_size += 1  # make it odd to preserve centering

        half_window = sample_window_size // 2

        padded_ts = np.pad(self.ts, (half_window, half_window), constant_values=(np.nan,))

        # moving average
        tmp = np.vstack(
            [padded_ts[i : i + sample_window_size] for i in range(len(padded_ts) - sample_window_size + 1)]
        )

        new_ts = np.nanmean(tmp, axis=1)

        self.ts = new_ts

        self._history.append(
            f"Smoothing has been applied with a window size of {window_size}"
        )

    def convert_seconds_to_samples(self, array):
        # this could probably be done in a more simple way?
        array_shape = array.shape
        flat_array = array.flatten()

        flat_array = np.array([int(second*self.fs) for second in flat_array])
                
        return flat_array.reshape(array_shape)

    def annotate_bad_segments(self, segments:np.ndarray, unit:Literal["s", "sample"]):
        """
        Annotates bad segments in the signal.
        Args:
            segments (np.ndarray): Array of segments to annotate. Dimensions should be (n, 2) where n is the number of segments. Each segment is defined by a start and end point.
            unit (str): Unit of the segments, either "s" for seconds or "sample" for samples.
        """
        if unit == "sample":
            # assume segments are already in samples
            segments = np.asarray(segments, dtype=int)

        elif unit== "s":
            segments = self.convert_seconds_to_samples(segments)
        
        else: 
            raise ValueError("Only accepts 's' or 'sample' as unit")
        
        # Validate segments
        if segments.ndim != 2 or segments.shape[1] != 2:
            raise ValueError("Segments must be a 2D array with shape (n, 2)")
        
        # check if segments are within bounds
        if np.any(segments < 0) or np.any(segments[:, 1]
                     > len(self.ts)):
                raise ValueError("Segments must be within the bounds of the time series")
        
        # check if start is less than end
        if np.any(segments[:, 0] >= segments[:, 1]):
            raise ValueError("Start of segment must be less than end of segment")
        
        self.bad_segments = segments

    @staticmethod
    def _peak_finder(ts, distance, prominence):

        peaks, _ = signal.find_peaks(ts, distance=distance, prominence=prominence)
        return peaks

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

    def phase_linear(self, peak_finder=None, distance=1, prominence=0.01):
        """
        Extract phase using linear interpolation between peaks and troughs.

        Parameters
        ----------
        peak_finder : callable, optional
            Optional custom function of the form ``func(ts, **kwargs) -> np.ndarray`` returning peak indices.
        distance : int
            Minimum distance in seconds between peaks, passed to the peak detection algorithm.
        prominence : float
            Prominence threshold for peak detection.

        Returns
        -------
        phase : np.ndarray
            The extracted phase values (in radians).
        peaks : np.ndarray
            Indices of detected peaks.
        troughs : np.ndarray
            Indices of detected troughs.
        """

        if peak_finder is None:
            distance = int(distance * self.fs)  # convert seconds to samples
            peaks = self._peak_finder(self.ts, distance=distance, prominence=prominence)
        else:
            try:
                peaks = peak_finder(self.ts)
            except Exception as e:
                raise RuntimeError(f"Custom peak_finder function failed: {e}")

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

        return (phase, peaks, troughs)

    def phase_onepoint(self, peak_finder=None, distance=100, prominence=0.01):
        """
        Extract phase by linearly interpolating from 0 to 2π between detected peaks.

        Parameters
        ----------
        peak_finder : callable, optional
            Optional custom function of the form ``func(ts, **kwargs) -> np.ndarray`` returning peak indices.
        distance : int
            Minimum distance between peaks, passed to the peak detection algorithm.
        prominence : float
            Prominence threshold for peak detection.

        Returns
        -------
        phase : np.ndarray
            The interpolated phase values (ranging from 0 to 2π).
        peaks : np.ndarray
            Indices of the detected peaks used for phase interpolation.
        """

        if peak_finder is None:
            peaks = self._peak_finder(self.ts, distance=distance, prominence=prominence)
        else:
            peaks = peak_finder(self.ts)  # assume user handles their own params

        phase = np.full_like(self.ts, np.nan, dtype=np.float32)

        for p1, p2 in zip(peaks[:-1], peaks[1:]):
            phase[p1:p2] = np.linspace(0, 2 * np.pi, p2 - p1)
            phase[p1] = 0
            phase[p2] = 2 * np.pi

        return phase, peaks

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

        Parameters
        ----------
        peak_finder : callable, optional
            Optional custom function of the form ``func(ts, **kwargs) -> np.ndarray`` returning peak indices.
        distance : int
            Minimum distance between peaks, used if no custom peak finder is provided.
        prominence : float
            Prominence threshold for peak detection.
        percentile : float
            Percentile of the absolute gradient below which a region is considered flat.
        descent_window : int
            Number of samples used to confirm descending and ascending slopes before and after flat regions.

        Returns
        -------
        phase : np.ndarray
            Phase array in radians (0 to 2π).
        peaks : np.ndarray
            Indices of detected peaks.
        troughs : list of tuple
            Each tuple contains the (start_index, end_index) of a flat region between peaks.
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
