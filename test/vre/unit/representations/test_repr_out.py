import numpy as np
import pytest
from vre.utils import MemoryData
from vre.representations import ReprOut

def test_ReprOut_ctor():
    with pytest.raises(AssertionError): # need to wrap output in MemoryData
        _ = ReprOut(frames=None, key=[1, 2, 3], output=np.array([1, 2, 3]))

    r1 = ReprOut(frames=None, key=[1, 2, 3], output=MemoryData(np.array([1, 2, 3])))
    assert (r1.output == [1, 2, 3]).all()

def test_ReprOut_equals():
    r1 = ReprOut(frames=None, key=[1, 2, 3], output=MemoryData(np.array([1, 2, 3])))
    r2 = ReprOut(frames=None, key=[1, 2, 3], output=MemoryData(np.array([1, 2, 3])))
    assert r1 == r2

    r3 = ReprOut(frames=None, key=[1, 2, 3], output=MemoryData(np.array([1, 3, 2])))
    assert r1 != r3

    r4 = ReprOut(frames=None, key=[1, 2, 3], output=MemoryData(np.array([1, 2, 3])), output_images=np.array([1,2,3]))
    assert r1 != r4

    r5 = ReprOut(frames=None, key=[1, 3, 2], output=MemoryData(np.array([1, 2, 3])))
    assert r1 != r5

def test_ReprOut_bad_key():
    with pytest.raises(AssertionError):
        _ = ReprOut(frames=None, key=[1, 2], output=MemoryData(np.array([1, 2, 3])))
    ReprOut(frames=None, key=[1, 2, 5], output=MemoryData(np.array([1, 2, 3])))
