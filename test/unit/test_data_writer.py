from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pytest
from vre.data_writer import DataWriter
from vre import ReprOut

def test_DataWriter_ctor_and_basic_writes():
    tmp_dir = Path(TemporaryDirectory().name)
    writer = DataWriter(tmp_dir, "rgb", output_dir_exists_mode="skip_computed",
                        binary_format="npz", image_format="png", compress=True)

    y_repr = ReprOut(np.random.randn(3, 240, 240))
    imgs = np.random.randint(0, 255, size=(3, 240, 240, 3)).astype(np.uint8)
    with pytest.raises(AssertionError):
        writer.write(y_repr, None, l=0, r=3)
    with pytest.raises(AssertionError):
        writer.write(y_repr, imgs.astype(np.int64), l=0, r=3)
    writer.write(y_repr, imgs, l=0, r=3)
    assert Path(tmp_dir / "rgb/npz/0.npz").exists()
    writer.write(y_repr, imgs.astype(np.uint32), l=0, r=3)
    with pytest.raises(AssertionError):
        writer.write(y_repr, imgs, l=0, r=2)

def test_DataWriter_all_batches_exist():
    tmp_dir = Path(TemporaryDirectory().name)
    writer = DataWriter(tmp_dir, "rgb", output_dir_exists_mode="skip_computed",
                        binary_format="npz", image_format=None, compress=True)

    y_repr = ReprOut(np.random.randn(3, 240, 240))
    for l in range(0, 2):
        for r in range(l+1, 3):
            assert writer.all_batch_exists(l=l, r=r) is False, (l, r)
    writer.write(y_repr, None, l=0, r=3)
    for l in range(0, 2):
        for r in range(l+1, 3):
            assert writer.all_batch_exists(l=l, r=r), (l, r)
    assert writer.all_batch_exists(l=0, r=4) is False
    with pytest.raises(AssertionError):
        writer.all_batch_exists(l=-1, r=3)
    with pytest.raises(AssertionError):
        writer.all_batch_exists(l="asdf", r=3)
    with pytest.raises(AssertionError):
        writer.all_batch_exists(l=0, r="qq")
    with pytest.raises(AssertionError):
        writer.all_batch_exists(l=3, r=0)
