import numpy as np
import pytest
from pyriodic.circular import Circular


def test_valid_radian_data():
    data = np.linspace(0, 2 * np.pi, 100)
    circ = Circular(data, unit="radians")
    assert circ.unit == "radians"
    assert circ.data.shape == (100,)


def test_valid_degree_data():
    degrees = [0, 90, 180, 270, 360]
    circ = Circular(degrees, unit="degrees")
    expected = np.deg2rad(degrees)
    np.testing.assert_allclose(circ.data, expected)
    assert circ.unit == "degrees"


def test_label_length_mismatch():
    data = [0, 90, 180]
    labels = ["a", "b"]
    with pytest.raises(ValueError, match="Length of labels must match"):
        Circular(data, labels=labels, unit="degrees")


def test_degrees_out_of_range():
    bad_degrees = [0, 90, 370]  # 370 > 360
    with pytest.raises(ValueError, match="unit is set to 'degrees'"):
        Circular(bad_degrees, unit="degrees")


def test_invalid_unit():
    with pytest.raises(ValueError, match="Invalid unit"):
        Circular([0, 1, 2], unit="bananas")


def test_mean_in_degrees():
    data = [80, 90, 180, 270, 280]
    circ = Circular(data, unit="degrees")
    mean = circ.mean()
    assert 170 < mean < 190  # Should be near 180°


def test_from_multiple():
    c1 = Circular([0, 90], unit="degrees")
    c2 = Circular([180, 270], unit="degrees")
    merged = Circular.from_multiple([c1, c2], labels=["A", "B"])
    assert merged.unit == "degrees"
    assert len(merged.data) == 4
    assert set(merged.labels) == {"A", "B"}
