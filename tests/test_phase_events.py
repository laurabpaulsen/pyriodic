import numpy as np
import pytest
from pyriodic import Circular
from pyriodic.phase_events import create_phase_events


def test_create_phase_events_unit_handling():
    # Simulate a linear phase ramp over 0 to 2π (or 0 to 360°)
    n_samples = 100
    radians_ts = np.linspace(0, 2 * np.pi, n_samples)
    degrees_ts = np.rad2deg(radians_ts)

    # Event indices at known phase locations
    events = np.array([0, 25, 50, 75, 99])  # evenly spaced

    # --- Case 1: Radians input/output ---
    result_rad = create_phase_events(phase_ts=radians_ts, events=events, unit="radians")

    assert isinstance(result_rad, Circular)
    assert result_rad.unit == "radians"
    np.testing.assert_allclose(result_rad.data, radians_ts[events])

    # --- Case 2: Degrees input/output ---
    result_deg = create_phase_events(phase_ts=degrees_ts, events=events, unit="degrees")

    assert isinstance(result_deg, Circular)
    assert result_deg.unit == "degrees"

    # Internally stored data is always in radians
    expected_radians = np.deg2rad(degrees_ts[events])
    np.testing.assert_allclose(result_deg.data, expected_radians, rtol=1e-5)

    # --- Case 3: Invalid unit should raise error ---
    with pytest.raises(ValueError, match="unit must be either 'radians' or 'degrees'"):
        create_phase_events(phase_ts=radians_ts, events=events, unit="gradians")
