import numpy as np
from typing import Union


# -----------------------------------------------------------------------
# Math utilities

def normalize_array(arr: np.ndarray,
                    min_v: Union[float,
                                 None] = None,
                    max_v: Union[float,
                                 None] = None):
    """
    Normalize an array to a range of [0, 1] in-place.

    Even if this function changes the array in-place, the result should be always used
    since the a casted copy is returned.

    Args:
        arr (np.ndarray): The input array to be normalized. It will be modified if it is float.
        min_v (float, optional): The minimum value for normalization. If None, the minimum of the array is used.
        max_v (float, optional): The maximum value for normalization. If None, the maximum of the array is used.

    Returns:
        np.ndarray: The normalized array (same as input if float, otherwise a float array).
    """
    # Ensure array is float (may create a new array if arr is integer)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(float, copy=False)

    data_min = arr.min() if min_v is None else min_v
    data_max = arr.max() if max_v is None else max_v

    if data_max == data_min:
        arr.fill(0.0)
        return arr

    arr -= data_min
    arr /= (data_max - data_min)

    return arr
