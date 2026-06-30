import numpy as np

from typing import Literal, Union
from tqdm import tqdm
from .preproc import RawSignal


def _format_surrogate_output(ts, events=None, return_ts=False):
    """
    Parameters
    ----------
    ts : ndarray
        Shape (n_surrogate, n_samples)

    events : ndarray | None

    return_ts : bool
        Whether to additionally return the full surrogate time series.

    Returns
    -------
    ndarray
        Samples at events or full time series.

    tuple(ndarray, ndarray)
        (samples, full_ts) if return_ts=True and events is not None.
    """
    
    if events is None:
        return ts


    events = np.asarray(events, dtype=np.intp)
    samples = np.take(ts, events, axis=1)
    #samples = ts[:, events]

    if return_ts:
        return samples, ts

    return samples

def _rng_check(rng=None):
    if rng is None:
        return np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        return rng
    else:
        raise ValueError("rng must be None or a numpy random Generator instance.")

def surrogate_shuffle_breath_cycles(phase_pool, events=None, n_surrogate=1000, rng=None, verbose=False, return_ts=False):
    
    def _get_breathing_cycles(phase_ts): 
        """Identify breathing cycles based on phase transitions."""
        breath_cycles = [] # Identify indices where phase wraps from 2π to 0 
        cycle_starts = np.where(np.diff(phase_ts) < -np.pi)[0] + 1 # +1 to get the index of the new cycle start 

        # loop over cycle_starts
        for start, end in zip(cycle_starts, cycle_starts[1:]):
            # print(f"Cycle starts at index: {start}, phase value: {phase_ts[start]}")
            # check if there are any nan values in the cycle
            if np.any(np.isnan(phase_ts[start:end])):
                if verbose:
                    print(f"Cycle from {start} to {end} contains NaN values, ignoring.")
            else:
                breath_cycles.append(phase_ts[start:end])
        return breath_cycles

    def _make_scrambled_PA(cycle_array, cycle_boundaries, rng, target_length):
        """shuffle indices of cycles and slice."""
        n_cycles = len(cycle_boundaries)
        perm = rng.permutation(n_cycles)

        # Compute start/end indices from permuted order
        starts = np.concatenate([[0], cycle_boundaries[:-1]])
        ends = cycle_boundaries
        segments = [cycle_array[s:e] for s, e in zip(starts[perm], ends[perm])]

        scrambled = np.concatenate(segments)
        if len(scrambled) > target_length:
            scrambled = scrambled[:target_length]
        elif len(scrambled) < target_length:
            # If needed, repeat and truncate
            n_repeat = int(np.ceil(target_length / len(scrambled)))
            scrambled = np.tile(scrambled, n_repeat)[:target_length]

        return scrambled

    def _precompute_cycle_array(breath_cycles):
        """Concatenate all cycles into one array, return the array and segment indices."""
        cycle_array = np.concatenate(breath_cycles)
        cycle_boundaries = np.cumsum([len(c) for c in breath_cycles])
        return cycle_array, cycle_boundaries


    rng = _rng_check(rng)

    breath_cycles = _get_breathing_cycles(phase_pool)
    cycle_array, cycle_boundaries = _precompute_cycle_array(breath_cycles)

    
    surr_ts = np.empty((n_surrogate, len(phase_pool)))
    for i in tqdm(range(n_surrogate)):
        scrambled_PA = _make_scrambled_PA(cycle_array, cycle_boundaries, rng, len(phase_pool))
        surr_ts[i] = scrambled_PA

    print(f"Generated {n_surrogate} surrogate time series by shuffling breathing cycles.")
    # 
    return _format_surrogate_output(surr_ts, events, return_ts)

def surrogate_time_shifted(
        phase_pool, 
        events=None, 
        n_surrogate=1000, 
        rng=None, 
        return_ts=False
    ):
    rng = _rng_check(rng)

    print("Generating null samples with time shifts...")

    # get an integer shift for each null sample )(between 0 and the number of samples in PA)
    shifts = rng.integers(0, len(phase_pool), size=n_surrogate)
    surrogate_ts = np.array([np.roll(phase_pool, shift) for shift in shifts])


    return _format_surrogate_output(surrogate_ts, events=events, return_ts=return_ts)

def surrogate_random(
        phase_pool, 
        events=None, 
        n_surrogate=1000, 
        rng=None, 
        return_ts=False
    ):
    """
    Generate surrogate time series by randomly shuffling the original phase time series.
    Parameters
    ----------
    phase_pool : np.ndarray
        The original phase time series from which to generate surrogates.
    events : np.ndarray
        Indices of the events for which to sample the surrogate phases.
    n_surrogate : int
        The number of surrogate samples to generate.
    rng : np.random.Generator, optional
        A random number generator for reproducibility. If None, a new generator will be created.
    return_ts : bool, optional
        If True, return the surrogate time series. If False, only return the formatted output.
    """
    rng = _rng_check(rng)

    # shuffle the phase pool n surrogate times
    surrogate_ts = np.array([rng.permutation(phase_pool) for _ in range(n_surrogate)])


    return _format_surrogate_output(surrogate_ts, events=events, return_ts=return_ts)

    



def surrogate_iaaft(ts, n_surrogate=1000, tol_pc=5., verbose=True, maxiter=1E6, sorttype="quicksort", rng=None):
    """
    Returns iAAFT surrogates of given time series.

    Parameter
    ---------
    ts : numpy.ndarray, with shape (N,)
        Input time series for which IAAFT surrogates are to be estimated.

    n_surrogate : int
        Number of surrogates to be generated.
    tol_pc : float
        Tolerance (in percent) level which decides the extent to which the
        difference in the power spectrum of the surrogates to the original
        power spectrum is allowed (default = 5).
    verbose : bool
        Show progress bar (default = `True`).
    maxiter : int
        Maximum number of iterations before which the algorithm should
        converge. If the algorithm does not converge until this iteration
        number is reached, the while loop breaks.
    sorttype : string
        Type of sorting algorithm to be used when the amplitudes of the newly
        generated surrogate are to be adjusted to the original data. This
        argument is passed on to `numpy.argsort`. Options include: 'quicksort',
        'mergesort', 'heapsort', 'stable'. See `numpy.argsort` for further
        information. Note that although quick sort can be a bit faster than 
        merge sort or heap sort, it can, depending on the data, have worse case
        spends that are much slower.

    Returns
    -------
    surrogate_ts : numpy.ndarray, with shape (n_surrogate, n_ts)
        Array containing the IAAFT surrogates of `ts` such that each row of `surrogate_ts`
        is an individual surrogate time series.
    """
    rng = _rng_check(rng)
    n_ts = ts.shape[0]
    surrogate_ts = np.nan * np.ones((n_surrogate, n_ts))
    ii = np.arange(n_ts)

    # get the fft of the original array
    x_amp = np.abs(np.fft.fft(ts))
    x_srt = np.sort(ts)
    r_orig = np.argsort(ts)

    # loop over surrogate number
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating IAAFT surrogates ..."
    for k in tqdm(range(n_surrogate), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):

        # 1) Generate random shuffle of the data
        count = 0
        r_prev = rng.permutation(ii)
        r_curr = r_orig
        z_n = ts[r_prev]
        percent_unequal = 100.

        # core iterative loop
        while (percent_unequal > tol_pc) and (count < maxiter):
            r_prev = r_curr

            # 2) FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random
            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)

            # 3) rescale zk to the original distribution of x
            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.) / n_ts

            # 4) repeat until number of unequal entries between r_curr and 
            # r_prev is less than tol_pc percent
            count += 1
            #if verbose and count % 20 == 0:
            #    print(f"Iteration {count}: {percent_unequal:.4f}% of ranks changed.")

        if count >= (maxiter - 1):
            print("maximum number of iterations reached!")
        else:
            # L
            surrogate_ts[k] = np.real(z_n)


    return _format_surrogate_output(surrogate_ts, events=None)



def extract_phase_angles_surrogate_ts(
    surrogate_ts,
    events,
    fs,
    preproc_pipeline=None,
    prominence=0.5,
    distance=0.5,
    return_phase_ts=True
):  
    phase_angles = np.empty((surrogate_ts.shape[0], len(events)))
    if return_phase_ts:
        phase_ts = np.empty((surrogate_ts.shape[0], surrogate_ts.shape[1]))
    
    for i in range(surrogate_ts.shape[0]):
        raw_surrogate = RawSignal(surrogate_ts[i], fs=fs)
        if preproc_pipeline is not None:
            raw_surrogate = preproc_pipeline.apply(raw_surrogate)

        phase, peaks, troughs = raw_surrogate.phase_twopoint(prominence=prominence, distance=distance)

        phase_angles[i] = phase[events]
        if return_phase_ts:
            phase_ts[i,:] = phase

    if return_phase_ts:
        return phase_angles, phase_ts
    else:
        return phase_angles

def old_surrogate_iaaft(phase_pool, events, n_surrogate=1000, rng=None, max_iter=10000, tol_pc=5, return_ts=False, verbose=True):
    """Generate IAAFT surrogate samples. Heavily based on the python implementation by Bedartha Goswami. See https://github.com/mlcs/iaaft
    
    
    Parameters
    ----------
    phase_pool : np.ndarray
        The original phase time series from which to generate surrogates.
    events : np.ndarray
        Indices of the events for which to sample the surrogate phases.
    n_surrogate : int
        The number of surrogate samples to generate.
    rng : np.random.Generator, optional
        A random number generator for reproducibility. If None, a new generator will be created.
    max_iter : int, optional
        Maximum number of iterations for the IAAFT algorithm.
    tol_pc : float, optional
        Tolerance for convergence in terms of percentage change in the power spectrum.

    """
    rng = _rng_check(rng)

    # empty array to hold surrogate times series for each surrogate sample
    surr_ts = np.empty((n_surrogate, len(phase_pool)))

    x = phase_pool.copy()

    # get the fft of the original array
    x_amp = np.fft.fft(phase_pool)
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Estimating IAAFT surrogate time courses..."
 
    for i in tqdm(range(n_surrogate), bar_format=pb_fmt, desc=pb_desc,
                  disable=not verbose):
        count = 0
        percent_unequal = 100.

        r_prev = rng.permutation(len(phase_pool))
        r_curr = r_orig
        z_n = x[r_prev]
        

        while (percent_unequal > tol_pc) and (count < max_iter):
            # STEP 1: random shuffle of the data
            r_prev = r_curr

            # STEP 2: FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random

            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)



            # STEP 3
            r_curr = np.argsort(z_n, kind="quicksort")
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.) / len(phase_pool)

            
            # STEP 4: repeat until convergence
            count += 1
            if verbose:
                print(f"Iteration {count}: {percent_unequal:.4f}% of ranks changed.")
            if count >= max_iter-1:
                print(f"Warning: IAAFT surrogate generation did not converge after {max_iter} iterations.")
            
                
            surr_ts[i] = np.real(z_n)

    
    # sample the surrogate time series at the event indices
    surr_samples = surr_ts[:, events]

    if return_ts:
        return surr_samples, surr_ts
    else:
        return surr_samples

def surrogate_uniform(events: Union[np.ndarray, int], n_surrogate=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(events, int):
        n_events = events
    else:
        n_events = len(events)
    return rng.uniform(0, 2*np.pi, (n_surrogate, n_events))

def surrogate_samples(
    events: np.ndarray,
    phase_pool: np.ndarray | Literal["uniform"] = "uniform",
    surrogate_method: Literal["time_shift", "scramble_breath_cycles", "random_sampling", "IAAFT"] = "random_sampling",
    n_surrogate: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate surrogate null samples of circular phase angles using specified methods.
    """
    if random_state is None:
        random_state = np.random.default_rng()

    if surrogate_method == "time_shift":
        return surrogate_time_shifted(phase_pool, events, n_surrogate, random_state)
    elif surrogate_method == "scramble_breath_cycles":
        return surrogate_shuffle_breath_cycles(phase_pool, events, n_surrogate, random_state)
    elif surrogate_method == "random_sampling":
        return surrogate_random(phase_pool, len(events), n_surrogate, random_state)
    elif surrogate_method == "IAAFT":
        return surrogate_iaaft(phase_pool, events, n_surrogate, random_state)
    else:
        raise ValueError(f"Unknown surrogate_method: {surrogate_method}")
