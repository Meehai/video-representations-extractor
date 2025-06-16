from vre.utils import MemoryData, ReprOut
import numpy as np

def test_MemoryData_ctor():
    memory = MemoryData([1, 2, 3, 4, 5], dtype=np.int32)
    assert memory[0] == 1
    assert memory.tolist() == [1, 2, 3, 4, 5]
    assert memory.dtype == np.int32
    assert memory.size == 5
    assert memory.shape == (5, )

def test_MemoryData_share_memory():
    data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    memory = MemoryData(data)
    data[0] = 100
    assert memory[0] == 100

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
