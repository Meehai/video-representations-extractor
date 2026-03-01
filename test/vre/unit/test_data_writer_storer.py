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

def test_DataWriter_ctor(tmp_path: Path):
    rgb = FakeRepresentation("rgb", [], output_dtype="float32")
    with pytest.raises(AssertionError): # must set binary_format or image_format
        _ = DataWriter(tmp_path, rgb, output_dir_exists_mode="skip_computed")
    rgb.binary_format = "npz"
    writer = DataWriter(tmp_path, rgb, output_dir_exists_mode="skip_computed")
    assert writer.rep_out_dir.exists()

def test_DataWriter_write_png(tmp_path: Path):
    # set image_format but then at write() time, the ReprOut has no image data. Then, we store data.
    rgb = FakeRepresentation("rgb", [], output_dtype="float32")
    rgb.image_format = "png"
    writer = DataWriter(tmp_path, rgb, output_dir_exists_mode="skip_computed")

    imgs = np.random.randint(0, 255, size=(3, 240, 240, 3)).astype(np.uint8)
    output = MemoryData(np.random.randn(3, 240, 240))
    with pytest.raises(TypeError): # image_format is set
        writer.write(ReprOut(frames=None, output=output, output_images=None, key=[0, 1, 2]))
    with pytest.raises(AssertionError): # bad image dtype
        writer.write(ReprOut(frames=None, output=output, output_images=imgs.astype(np.int16), key=[0, 1, 2]))
    writer.write(ReprOut(frames=None, output=output, output_images=imgs, key=[0, 1, 2]))
    assert (tmp_path / "rgb/png/0.png").exists()
    assert (tmp_path / "rgb/png/1.png").exists()
    assert (tmp_path / "rgb/png/2.png").exists()
    assert not (tmp_path / "rgb/png/3.png").exists()

def test_DataWriter_write_npz(tmp_path: Path):
    # same as write_png, but with npz. First no data, then other stuff.
    rgb = FakeRepresentation("rgb", [], output_dtype="float32")
    rgb.binary_format = "npz"
    writer = DataWriter(tmp_path, rgb, output_dir_exists_mode="skip_computed")

    with pytest.raises(AssertionError):
        writer.write(ReprOut(frames=None, output=None, key=[0, 1, 2]))

    output = MemoryData(np.random.randn(3, 240, 240))
    writer.write(ReprOut(frames=None, output=output, key=[0, 1, 2]))
    assert (tmp_path / "rgb/npz/0.npz").exists()
    assert (tmp_path / "rgb/npz/1.npz").exists()
    assert (tmp_path / "rgb/npz/2.npz").exists()
    assert not (tmp_path / "rgb/npz/3.npz").exists()

def test_DataWriter_all_batches_exist():
    tmp_dir = Path(TemporaryDirectory().name)
    rgb = FakeRepresentation("rgb", [], output_dtype="float32")
    rgb.set_io_params(binary_format="npz")
    writer = DataWriter(tmp_dir, rgb, output_dir_exists_mode="skip_computed")

    y_repr = ReprOut(None, MemoryData(np.random.randn(3, 240, 240)), key=[0, 1, 2])
    for l in range(0, 2):
        for r in range(l+1, 3):
            assert writer.all_batch_exists(list(range(l, r))) is False, (l, r)
    writer.write(y_repr)
    for l in range(0, 2):
        for r in range(l+1, 3):
            assert writer.all_batch_exists(list(range(l, r))), (l, r)
    assert writer.all_batch_exists([0, 1, 2, 3]) is False
    with pytest.raises(AssertionError):
        writer.all_batch_exists([-1, 0, 1, 2])
    with pytest.raises(AssertionError):
        writer.all_batch_exists([1, "asdf", 3])

@pytest.mark.parametrize("n_threads", [0, 1, 4])
def test_DataStorer(n_threads: int):
    tmp_dir = Path(TemporaryDirectory().name)
    rgb, hsv = FakeRepresentation("rgb", [], output_dtype="float32"), FakeRepresentation("hsv", [], output_dtype="float32")
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
