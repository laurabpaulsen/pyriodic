import numpy as np
import pytest
from pyriodic.circular import Circular


def test_valid_radian_data():
    data = np.linspace(-np.pi, np.pi, 100)
    circ = Circular(data, unit="radians", zero=0)
    assert circ.unit == "radians"
    assert circ.zero == 0
    assert circ.data.shape == (100,)


def test_invalid_unit():
    with pytest.raises(ValueError, match="Invalid unit"):
        Circular([0, 1, 2], unit="bananas")


def test_invalid_data_type():
    with pytest.raises(ValueError, match="Invalid data_type"):
        Circular([0, 1, 2], data_type="something_else")


def test_invalid_zero():
    with pytest.raises(ValueError, match="Invalid zero"):
        Circular([0, 1, 2], zero="wrong")


def test_radians_mislabelled_as_degrees():
    # Data within 0–2π but labeled as degrees → should raise error
    data = np.linspace(0, 2 * np.pi, 100)
    with pytest.raises(ValueError, match="but unit is set to 'degrees'"):
        Circular(data, unit="degrees")


def test_degrees_mislabelled_as_radians():
    # Data in degrees range but labeled as radians → should raise error
    data = [0, 90, 180, 270]
    with pytest.raises(ValueError, match="exceed the valid radian range"):
        Circular(data, unit="radians")


def test_conversion_to_degrees():
    data = np.array([0, np.pi / 2, np.pi])
    circ = Circular(data, unit="radians")
    circ.convert_to("degrees")
    expected = np.array([0, 90, 180])
    np.testing.assert_allclose(circ.data, expected, atol=1e-6)


def test_conversion_to_radians():
    data = np.array([0, 90, 180])
    circ = Circular(data, unit="degrees")
    circ.convert_to("radians")
    expected = np.array([0, np.pi / 2, np.pi])
    np.testing.assert_allclose(circ.data, expected, atol=1e-6)
