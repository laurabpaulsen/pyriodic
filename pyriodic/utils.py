
from typing import Union
import numpy as np

def dat2rad(
        data: Union[np.ndarray, float, int],
        full_range:  Union[float, int] = 360
        ):
    
    return 2 * np.pi * data / full_range 


def rad2dat(
        rad: Union[np.ndarray, float, int],
        full_range:  Union[float, int] = 360
        ):
    
    return  full_range * rad / (2 * np.pi)