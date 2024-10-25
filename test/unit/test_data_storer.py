from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pytest
from vre.data_writer import DataWriter
from vre.data_storer import DataStorer
from vre import ReprOut

@pytest.mark.parametrize("n_threads", [0, 1, 4])
def test_DataStorer_ctor(n_threads: int):
    tmp_dir = Path(TemporaryDirectory().name)
    writer = DataWriter(tmp_dir, ["rgb", "hsv"], output_dir_exists_mode="skip_computed",
                        binary_format="npz", image_format="png", compress=True)
    storer = DataStorer(writer, n_threads=n_threads)

    y = np.random.randn(10, 240, 240)
    imgs = np.random.randint(0, 255, size=(10, 240, 240, 3)).astype(np.uint8)

    batches = [[0, 3], [3, 6], [6, 10]]
    for repres in writer.representations:
        for l, r in batches:
            print(l, r, imgs[l:r].shape)
            storer(repres, ReprOut(y[l:r]), imgs[l:r], l=l, r=r)
    storer.join_with_timeout(2)
    if n_threads > 0:
        with pytest.raises(AssertionError):
            storer(repres, ReprOut(y[0:1]), imgs[0:1], l=0, r=1)
    assert writer.all_batch_exists("rgb", l=0, r=10)
    assert writer.all_batch_exists("hsv", l=0, r=10)

if __name__ == "__main__":
    test_DataStorer_ctor(4)
