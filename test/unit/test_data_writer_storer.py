import sys
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pytest
from vre import ReprOut
from vre.utils import get_project_root
from vre.data_writer import DataWriter
from vre.data_storer import DataStorer

sys.path.append(str(get_project_root() / "test"))
from fake_representation import FakeRepresentation

def all_batch_exists(writer: DataWriter, l: int, r: int) -> bool:
    """true if all batch [l:r] exists on the disk"""
    def _path(writer: DataWriter, t: int, suffix: str) -> Path:
        return writer.output_dir / writer.rep.name / suffix / f"{t}.{suffix}"
    assert isinstance(l, int) and isinstance(r, int) and 0 <= l < r, (l, r, type(l), type(r))
    assert writer.rep.export_binary or writer.rep.export_image, writer.rep
    for ix in range(l, r):
        if writer.rep.export_binary and not _path(writer, ix, writer.rep.binary_format.value).exists():
            return False
        if writer.rep.export_image and not _path(writer, ix, writer.rep.image_format.value).exists():
            return False
    return True

def test_DataWriter_ctor_and_basic_writes():
    tmp_dir = Path(TemporaryDirectory().name)
    rgb = FakeRepresentation("rgb", [])
    rgb.set_compute_params(binary_format="npz", image_format="png")
    writer = DataWriter(tmp_dir, rgb, output_dir_exists_mode="skip_computed")

    y_repr = ReprOut(np.random.randn(3, 240, 240), key=[0, 1, 2])
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
    rgb = FakeRepresentation("rgb", [])
    rgb.set_compute_params(binary_format="npz")
    writer = DataWriter(tmp_dir, rgb, output_dir_exists_mode="skip_computed")

    y_repr = ReprOut(np.random.randn(3, 240, 240), key=[0, 1, 2])
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

@pytest.mark.parametrize("n_threads", [0, 1, 4])
def test_DataStorer(n_threads: int):
    tmp_dir = Path(TemporaryDirectory().name)
    rgb, hsv = FakeRepresentation("rgb", []), FakeRepresentation("hsv", [])
    rgb.set_compute_params(binary_format="npz", image_format="png")
    hsv.set_compute_params(binary_format="npz", image_format="jpg")
    writer_rgb = DataWriter(tmp_dir, rgb, output_dir_exists_mode="skip_computed")
    writer_hsv = DataWriter(tmp_dir, hsv, output_dir_exists_mode="skip_computed")
    storers = [DataStorer(writer_rgb, n_threads=n_threads), DataStorer(writer_hsv, n_threads=n_threads)]

    y = np.random.randn(10, 240, 240)
    imgs = np.random.randint(0, 255, size=(10, 240, 240, 3)).astype(np.uint8)

    batches = [[0, 3], [3, 6], [6, 10]]
    for storer in storers:
        for l, r in batches:
            storer(ReprOut(y[l:r], key=slice(l, r)), imgs[l:r], l=l, r=r)
    [storer.join_with_timeout(2) for storer in storers]
    if n_threads > 0:
        with pytest.raises(AssertionError):
            storer(ReprOut(y[0:1], key=[0]), imgs[0:1], l=0, r=1)
    assert all_batch_exists(writer_rgb, l=0, r=10)
    assert all_batch_exists(writer_hsv, l=0, r=10)
