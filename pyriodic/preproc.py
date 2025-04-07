from circular import Circular
import numpy as np
from scipy import interpolate, signal
from scipy.stats import gaussian_kde
from scipy.signal import hilbert, butter, filtfilt
import logging
logger = logging.getLogger(__name__)

def aaaaa(data) -> Circular: 
    pass

def preprocesses_ts(ts, nan_threshold = 2.5):
     # set outliers to NaN based on z-score threshold 
    z_scores = np.abs((ts - np.nanmean(ts)) / np.nanstd(ts))
    logger.info(f"Found {sum(z_scores > 2.5)} outliers")
    ts[z_scores > nan_threshold] = np.nan


    # linear interpolation of outlier segments
    nans = np.isnan(ts)
    x_vals = np.where(~nans)[0]
    y_vals = ts[~nans]
    interpolated_values = interpolate.interp1d(x_vals, y_vals, kind='linear', fill_value="extrapolate")
    ts[nans] = interpolated_values(np.where(nans)[0])
    logger.info("Linear interpolation of NaN finished")


    # normalise interpolated time series
    normalised_ts = (ts - np.nanmean(ts)) / np.nanstd(ts)

    return normalised_ts


def extract_peaks_and_troughs(ts, widths = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts peaks and troughs from timeseries data
    
    Parameters:
    ts (np.array): An array of timeseries datapoints

    Returns:

    """

    # POTENTIAL OPTIMISATION OF FUNCTION
    # auto for widths. Is there some way to programmatically make a good guess about the width for peak detection?


    # finding peaks and troughs
    logger.info("Looking for peaks - this may take a while")
    peaks = signal.find_peaks_cwt(ts, widths = widths)#, distance = min_sample)
    if len(peaks)==0:
        raise ValueError(f"No peaks were identified. Consider lowering the widths parameter which is currently set to {widths}")
    # old way of doing it -> peaks = signal.find_peaks(normalised_ts)[0] figure out what works best on data!!

    logger.info("Done looking for peaks")
    troughs = np.array([], dtype = int)

    for peak1, peak2 in zip(peaks, peaks[1:]):                  # finding the troughs -> the minimum between the peaks
        tmp_resp = ts[peak1:peak2]                   # respiration time course between the peaks

        # DIFFERENT WAYS OF FINDING TROUGH

        # take the index in the middle between the two peaks
        #trough_ind = peak2 - peak1

        # finding minimum between two peaks
        trough_ind_tmp = np.where(tmp_resp == min(tmp_resp))
        try:
            trough_ind = int(trough_ind_tmp[0] + peak1)                # get the index relative to the entire time series
        except TypeError:
            trough_ind = int(np.mean(trough_ind_tmp[0]) + peak1) 


        # other ways??


        troughs = np.append(troughs, trough_ind) 
    

    return peaks, troughs

def extract_phase_angle(ts, method="linear", **kwargs):
    if method == "linear":
        return extract_phase_angle_linear(ts, **kwargs)
    elif method == "hilbert":
        return extract_phase_angle_hilbert(ts, **kwargs)
    
    else:
        raise NotImplementedError

def extract_phase_angle_linear(ts, nan_threshold = 2.5, widths = 500):
    # detect outliers, interpolate and normalise timeseries data
    normalised_ts = preprocesses_ts(ts, nan_threshold = nan_threshold)

    peaks, troughs = extract_peaks_and_troughs(normalised_ts, widths=widths)
    print(peaks)

    # linear interpolation between peaks and troughs to get phase angle
    phase_angle = np.zeros(len(normalised_ts))
    phase_angle[:] = np.nan # fill with nans
        
    # set troughs to pi and peaks to 0
    phase_angle[troughs], phase_angle[peaks] = np.pi, 0

    # interpolate the phase angle between peaks and troughs
    for peak1, peak2, trough in zip(peaks, peaks[1:], troughs):
        phase_angle[peak1:trough] = np.linspace(0 + np.pi/(trough-peak1), np.pi,  trough-peak1)
        phase_angle[trough:peak2] = np.linspace(-np.pi + np.pi/(peak2-trough), 0, peak2-trough)

    return phase_angle, peaks, troughs



def extract_phase_angle_hilbert(ts, fs=1.0, bandpass=(0.05, 0.5)):
    """
    Extract instantaneous phase angle using the Hilbert Transform.

    Parameters:
        ts (np.ndarray): 1D time series
        fs (float): sampling frequency (Hz)
        bandpass (tuple): bandpass filter range (low, high) in Hz

    Returns:
        np.ndarray: phase angles in radians (-π to π)
    """
    if np.isnan(ts).any():
        ts = preprocesses_ts(ts)


    # Optional: bandpass filter
    nyq = 0.5 * fs
    low, high = bandpass[0] / nyq, bandpass[1] / nyq
    b, a = butter(2, [low, high], btype='band')

    filtered = filtfilt(b, a, ts)

    # Hilbert transform
    analytic_signal = hilbert(filtered)
    phase_angles = np.angle(analytic_signal)  # returns in range [-π, π]

    return phase_angles