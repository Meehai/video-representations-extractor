from vre_repository.optical_flow.optical_flow_representation import _get_delta_frames
import numpy as np
import pytest

def test_get_delta_frames():
    assert _get_delta_frames(np.zeros(100), ixs=[0, 5], delta=1) == [1, 6]
    assert _get_delta_frames(np.zeros(100), ixs=[0, 100], delta=9) == [9, 99]
    assert _get_delta_frames(np.zeros(100), ixs=[0, 100], delta=-9) == [0, 91]
    assert _get_delta_frames(np.zeros(50), ixs=[0, 100], delta=9) == [9, 49]
    with pytest.raises(AssertionError):
        _ = _get_delta_frames(np.zeros(100), ixs=[0, 100], delta=0)
