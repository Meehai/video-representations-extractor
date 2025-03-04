from vre_repository.optical_flow.optical_flow_representation import _get_delta_frames
import numpy as np

def test_get_delta_frames():
    vid = np.zeros(100)
    assert _get_delta_frames(vid, ixs=[0, 5], delta=1) == [1, 6]
