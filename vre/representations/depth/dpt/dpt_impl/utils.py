# pylint: disable=all
import numpy as np

def _constrain_to_multiple_of(x: int, multiple_of: int, min_val=0, max_val=None) -> int:
    x = np.array(x)
    y: np.ndarray = (np.round(x / multiple_of) * multiple_of).astype(int)
    if max_val is not None and y > max_val:
        y = (np.floor(x / multiple_of) * multiple_of).astype(int)
    if y < min_val:
        y = (np.ceil(x / multiple_of) * multiple_of).astype(int)
    return y.item()

def get_size(__height, __width, height, width, multiple_of) -> tuple[int, int]:
    # determine new height and width
    scale_height = __height / height
    scale_width = __width / width
    # keep aspect ratio
    if abs(1 - scale_width) < abs(1 - scale_height):
        # fit width
        scale_height = scale_width
    else:
        # fit height
        scale_width = scale_height
    new_height = _constrain_to_multiple_of(scale_height * height, multiple_of)
    new_width = _constrain_to_multiple_of(scale_width * width, multiple_of)

    return new_height, new_width
