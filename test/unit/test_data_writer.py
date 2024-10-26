from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pytest
from vre.data_writer import DataWriter
from vre import ReprOut

def all_batch_exists(writer: DataWriter, l: int, r: int) -> bool:
    """true if all batch [l:r] exists on the disk"""
    assert isinstance(l, int) and isinstance(r, int) and 0 <= l < r, (l, r, type(l), type(r))
    for ix in range(l, r):
        if writer.export_binary and not writer._path(ix, writer.binary_format).exists():
            return False
        if writer.export_image and not writer._path(ix, writer.image_format).exists():
            return False
    return True

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
            assert all_batch_exists(writer, l=l, r=r) is False, (l, r)
    writer.write(y_repr, None, l=0, r=3)
    for l in range(0, 2):
        for r in range(l+1, 3):
            assert all_batch_exists(writer, l=l, r=r), (l, r)
    assert all_batch_exists(writer, l=0, r=4) is False
    with pytest.raises(AssertionError):
        all_batch_exists(writer, l=-1, r=3)
    with pytest.raises(AssertionError):
        all_batch_exists(writer, l="asdf", r=3)
    with pytest.raises(AssertionError):
        all_batch_exists(writer, l=0, r="qq")
    with pytest.raises(AssertionError):
        all_batch_exists(writer, l=3, r=0)
