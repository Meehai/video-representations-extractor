from tempfile import TemporaryDirectory
import numpy as np
import pytest
from vre_video.readers import NumpyFrameReader

@pytest.fixture
def tmpdir(request: pytest.FixtureRequest):
    tmpdir = TemporaryDirectory()
    request.addfinalizer(lambda: tmpdir.cleanup())
    yield tmpdir.name

def test_NumpyFrameReader_ctor_npy_dir(tmpdir: str):
    for i in range(10):
        np.save(f"{tmpdir}/{i}.npy", np.zeros((30, 30, 3), dtype=np.uint8))

    reader = NumpyFrameReader(tmpdir)
    assert reader.shape == (10, 30, 30, 3)
    assert reader.fps == 1  # default
    assert reader[0].shape == (30, 30, 3)
    assert reader[0:2].shape == (2, 30, 30, 3)
    assert reader[[0, 5]].shape == (2, 30, 30, 3)
    assert reader[np.array([0, 5])].shape == (2, 30, 30, 3)

def test_NumpyFrameReader_ctor_npz_dir(tmpdir: str):
    for i in range(10):
        np.savez(f"{tmpdir}/{i}.npz", np.zeros((30, 30, 3), dtype=np.uint8))

    reader = NumpyFrameReader(tmpdir)
    assert reader.shape == (10, 30, 30, 3)

def test_NumpyFrameReader_ctor_from_list():
    data = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(10)]
    reader = NumpyFrameReader(data)
    assert reader.shape == (10, 32, 32, 3)

def test_NumpyFrameReader_ctor_from_array():
    arr = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    reader = NumpyFrameReader(arr)
    assert reader.shape == (10, 32, 32, 3)

def test_NumpyFrameReader_bad_shape_list():
    data = [np.zeros((30 + i, 30, 3), dtype=np.uint8) for i in range(10)]
    with pytest.raises(AssertionError):
        _ = NumpyFrameReader(data)

def test_NumpyFrameReader_bad_dtype_list():
    data = [np.zeros((30, 30, 3), dtype=np.float32) for _ in range(10)]
    with pytest.raises(AssertionError):
        _ = NumpyFrameReader(data)

def test_NumpyFrameReader_bad_dtype_array():
    arr = np.zeros((10, 30, 30, 3), dtype=np.float32)
    with pytest.raises(AssertionError):
        _ = NumpyFrameReader(arr)

def test_NumpyFrameReader_mixed_extensions(tmpdir: str):
    np.save(f"{tmpdir}/0.npy", np.zeros((30, 30, 3), dtype=np.uint8))
    np.savez(f"{tmpdir}/1.npz", np.zeros((30, 30, 3), dtype=np.uint8))
    with pytest.raises(AssertionError):
        _ = NumpyFrameReader(tmpdir)

def test_NumpyFrameReader_ctor_from_array_frames():
    arr = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    reader = NumpyFrameReader(arr, frames=list(range(10, 20)))

    assert len(reader) == 20
    with pytest.raises(KeyError):
        _ = reader[0]
    assert reader.frames == list(range(10, 20))
    assert reader[10].shape == (32, 32, 3)
    assert reader[[10, 15, 18]].shape == (3, 32, 32, 3)

def test_NumpyFrameReader_ctor_npz_dir_frames(tmpdir: str):
    for i in range(10):
        np.savez(f"{tmpdir}/{i+10}.npz", np.zeros((30, 30, 3), dtype=np.uint8))

    reader = NumpyFrameReader(tmpdir)
    assert reader.frames == list(range(10, 20))
    assert reader.shape == (10, 30, 30, 3)
    assert reader[10].shape == (30, 30, 3)
    assert reader[[10, 15, 18]].shape == (3, 30, 30, 3)
