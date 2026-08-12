import numpy as np
import sys

def fn(frame: np.ndarray):
    print("hello", file=sys.stderr)
    left_side = frame[:, 0:frame.shape[1] // 2]
    right_side = frame[:, frame.shape[1] // 2:]
    right_side = right_side[::-1]
    frame = np.concatenate([right_side, left_side], axis=1)
    return frame
