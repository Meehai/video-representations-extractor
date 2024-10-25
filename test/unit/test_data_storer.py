from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pytest
from vre.data_writer import DataWriter
from vre.data_storer import DataStorer
from vre import ReprOut

@pytest.mark.parametrize("n_threads", [0, 1, 4])
def test_DataStorer(n_threads: int):
    tmp_dir = Path(TemporaryDirectory().name)
    writer_rgb = DataWriter(tmp_dir, "rgb", output_dir_exists_mode="skip_computed",
                            binary_format="npz", image_format="png", compress=True)
    writer_hsv = DataWriter(tmp_dir, "hsv", output_dir_exists_mode="skip_computed",
                            binary_format="npz", image_format="png", compress=True)
    storers = [DataStorer(writer_rgb, n_threads=n_threads), DataStorer(writer_hsv, n_threads=n_threads)]

    y = np.random.randn(10, 240, 240)
    imgs = np.random.randint(0, 255, size=(10, 240, 240, 3)).astype(np.uint8)

    batches = [[0, 3], [3, 6], [6, 10]]
    for storer in storers:
        for l, r in batches:
            storer(ReprOut(y[l:r]), imgs[l:r], l=l, r=r)
    [storer.join_with_timeout(2) for storer in storers]
    if n_threads > 0:
        with pytest.raises(AssertionError):
            storer(ReprOut(y[0:1]), imgs[0:1], l=0, r=1)
    assert writer_rgb.all_batch_exists(l=0, r=10)
    assert writer_hsv.all_batch_exists(l=0, r=10)
