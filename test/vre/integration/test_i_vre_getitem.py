import sys
import numpy as np
import pytest
from vre import VRE
from vre.utils import get_project_root
from vre.vre_video import FakeVideo

sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation

@pytest.fixture
def video() -> FakeVideo:
    return FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), fps=30)

def test_vre_getitem_basic(video: FakeVideo):
    vre = VRE(video=video, representations=[rgb := FakeRepresentation("rgb", n_channels=3),
                                            FakeRepresentation("hsv", n_channels=3, dependencies=[rgb])])
    vre.set_io_parameters(binary_format="npz")
    with pytest.raises(KeyError):
        _ = vre[2]
    res = vre[0]
    assert res.keys() == {"rgb", "hsv"}
    assert res["rgb"].output.shape == (1, 128, 128, 3)
    assert res["rgb"].output_images is None
    assert res["hsv"].output.shape == (1, 128, 128, 3)
    assert res["hsv"].output_images is None

    vre.set_io_parameters(binary_format="npz", image_format="png")
    res = vre[0]
    assert res.keys() == {"rgb", "hsv"}
    assert res["rgb"].output.shape == (1, 128, 128, 3)
    assert res["rgb"].output_images.shape == (1, 128, 128, 3)
    assert res["hsv"].output.shape == (1, 128, 128, 3)
    assert res["hsv"].output_images.shape == (1, 128, 128, 3)

def test_vre_getitem_batch(video: FakeVideo):
    vre = VRE(video=video, representations=[rgb := FakeRepresentation("rgb", n_channels=3),
                                            FakeRepresentation("hsv", n_channels=3, dependencies=[rgb])])
    vre.set_io_parameters(binary_format="npz", image_format="png")
    res = vre[0:2]
    assert res.keys() == {"rgb", "hsv"}
    assert res["rgb"].output.shape == (2, 128, 128, 3)
    assert res["rgb"].output_images.shape == (2, 128, 128, 3)
    assert res["hsv"].output.shape == (2, 128, 128, 3)
    assert res["hsv"].output_images.shape == (2, 128, 128, 3)

def test_vre_getitem_resolution(video: FakeVideo):
    vre = VRE(video=video, representations=[rgb := FakeRepresentation("rgb", n_channels=3),
                                            FakeRepresentation("hsv", n_channels=3, dependencies=[rgb])])
    vre.set_io_parameters(binary_format="npz", image_format="png", output_size=(64, 64))
    res = vre[0:2]
    assert res.keys() == {"rgb", "hsv"}
    assert res["rgb"].output.shape == (2, 64, 64, 3)
    assert res["rgb"].output_images.shape == (2, 64, 64, 3)
    assert res["hsv"].output.shape == (2, 64, 64, 3)
    assert res["hsv"].output_images.shape == (2, 64, 64, 3)
