import numpy as np

from typing import Literal

 




def surrogate_scramble_breath_cycles(phase_pool, events, n_surrogate, rng = None):
    
    def _get_breathing_cycles(phase_ts): 
        """Identify breathing cycles based on phase transitions."""
        breath_cycles = [] # Identify indices where phase wraps from 2π to 0 
        cycle_starts = np.where(np.diff(phase_ts) < -np.pi)[0] + 1 # +1 to get the index of the new cycle start 

        # loop over cycle_starts
        for start, end in zip(cycle_starts, cycle_starts[1:]):
            # print(f"Cycle starts at index: {start}, phase value: {phase_ts[start]}")
            # check if there are any nan values in the cycle
            if np.any(np.isnan(phase_ts[start:end])):
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




    if rng is None:
        rng = np.random.default_rng()

    breath_cycles = _get_breathing_cycles(phase_pool)
    cycle_array, cycle_boundaries = _precompute_cycle_array(breath_cycles)

    surr_samples = []
    for _ in range(n_surrogate):
        scrambled_PA = _make_scrambled_PA(cycle_array, cycle_boundaries, rng, len(phase_pool))
        surr_samples.append(scrambled_PA[events])

    return np.array(surr_samples)

def generate_time_shifted_samples(phase_pool, events, n_surrogate, rng=None):
    if rng is None:
        rng = np.random.default_rng()


    print("Generating null samples with time shifts...")
    surr_samples = []

    # get an integer shift for each null sample )(between 0 and the number of samples in PA)
    shifts = rng.integers(0, len(phase_pool), size=n_surrogate)
    
    for shift in shifts:
        # shift PA by the random amount
        shifted = np.roll(phase_pool, shift)
        # sample the same number of phases as in the observed data
        surr_sample = shifted[events]
        surr_samples.append(surr_sample)

    return np.array(surr_samples)

def generate_random_samples(phase_pool, n_events, n_surrogate, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    null_samples = [
        rng.choice(phase_pool, size=n_events, replace=False) for _ in range(n_surrogate)
    ]

    return np.array(null_samples)

def generate_iaaft_samples(phase_pool, events, n_surrogate, rng=None):
    """Generate IAAFT surrogate samples. Heavily based on the python implementation by Bedartha Goswami. See https://github.com/mlcs/iaaft"""
    if rng is None:
        rng = np.random.default_rng()
    raise NotImplementedError("IAAFT surrogate generation is not yet implemented.")

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
        return generate_time_shifted_samples(phase_pool, events, n_surrogate, random_state)
    elif surrogate_method == "scramble_breath_cycles":
        return surrogate_scramble_breath_cycles(phase_pool, events, n_surrogate, random_state)
    elif surrogate_method == "random_sampling":
        return generate_random_samples(phase_pool, len(events), n_surrogate, random_state)
    elif surrogate_method == "IAAFT":
        return generate_iaaft_samples(phase_pool, events, n_surrogate, random_state)
    else:
        raise ValueError(f"Unknown surrogate_method: {surrogate_method}")
