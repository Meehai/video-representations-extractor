import sys
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pytest
from vre import ReprOut
from vre.utils import get_project_root, MemoryData
from vre.data_writer import DataWriter
from vre.data_storer import DataStorer

sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation

@pytest.mark.parametrize("n_threads", [0, 1, 4])
def test_DataStorer(n_threads: int):
    tmp_dir = Path(TemporaryDirectory().name)
    rgb = FakeRepresentation("rgb", [], output_dtype="float32")
    hsv = FakeRepresentation("hsv", [], output_dtype="float32")
    rgb.set_io_params(binary_format="npz", image_format="png")
    hsv.set_io_params(binary_format="npz", image_format="jpg")
    writer_rgb = DataWriter(tmp_dir, rgb, output_dir_exists_mode="skip_computed")
    writer_hsv = DataWriter(tmp_dir, hsv, output_dir_exists_mode="skip_computed")
    storers = [DataStorer(writer_rgb, n_threads=n_threads), DataStorer(writer_hsv, n_threads=n_threads)]

    y = np.random.randn(10, 240, 240)
    imgs = np.random.randint(0, 255, size=(10, 240, 240, 3)).astype(np.uint8)

    batches = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    for storer in storers:
        for batch in batches:
            storer(ReprOut(frames=None, output=MemoryData(y[batch]), output_images=imgs[batch], key=batch))
    [storer.join_with_timeout(2) for storer in storers]
    if n_threads > 0:
        with pytest.raises(AssertionError):
            storer(ReprOut(frames=None, output=MemoryData(y[0:1]), output_images=imgs[0:1], key=[0]))
    assert writer_rgb.all_batch_exists(list(range(10)))
    assert writer_hsv.all_batch_exists(list(range(10)))
