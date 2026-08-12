from tempfile import TemporaryDirectory
import numpy as np
import pytest
from vre_video.utils import image_write
from vre_video.readers import PILFrameReader

@pytest.fixture
def tmpdir(request: pytest.FixtureRequest):
    tmpdir = TemporaryDirectory()
    request.addfinalizer(lambda: tmpdir.cleanup())
    yield tmpdir.name

def test_PILFrameReader_ctor(tmpdir: str):
    for i in range(10):
        image_write(np.zeros((30, 30, 3), dtype=np.uint8), f"{tmpdir}/{i}.png")

    reader = PILFrameReader(tmpdir)
    assert reader.shape == (10, 30, 30, 3)
    assert reader.fps == 1 # default
    assert reader[0].shape == (30, 30, 3)
    assert reader[0:2].shape == (2, 30, 30, 3)
    assert reader[[0, 5]].shape == (2, 30, 30, 3)
    assert reader[np.array([0, 5])].shape == (2, 30, 30, 3)

def test_PILFrameReader_bad_shape(tmpdir: str):
    # not all the underlying frames are equally sized
    for i in range(10):
        image_write(np.zeros((30 + i, 30, 3), dtype=np.float32), f"{tmpdir}/{i}.png")
    with pytest.raises(AssertionError):
        _ = PILFrameReader(tmpdir)
