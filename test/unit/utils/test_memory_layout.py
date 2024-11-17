from vre.utils import MemoryData
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
