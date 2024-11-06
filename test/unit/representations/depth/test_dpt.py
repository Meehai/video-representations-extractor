import numpy as np
from vre.representations.depth.dpt import DepthDpt
from vre.utils import FakeVideo

def test_dpt():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    dpt_repr = DepthDpt(name="dpt", dependencies=[])
    dpt_repr.vre_setup(load_weights=False)
    assert dpt_repr.name == "dpt"
    assert dpt_repr.compress is True # default from ComputeRepresentationMixin
    assert dpt_repr.device == "cpu" # default from LearnedRepresentationMixin

    frames = np.array(video[0:1])
    y_dpt = dpt_repr(frames)
    assert y_dpt.output.shape == (1, 192, 384), y_dpt.output.shape

    y_dpt_images = dpt_repr.make_images(frames, y_dpt)
    assert y_dpt_images.shape == (1, 192, 384, 3), y_dpt_images.shape
    assert y_dpt_images.dtype == np.uint8, y_dpt_images.dtype

    assert dpt_repr.size(y_dpt) == (192, 384)
    y_normals_resized = dpt_repr.resize(y_dpt, (64, 128)) # we can resize it though
    assert dpt_repr.size(y_normals_resized) == (64, 128)
    assert dpt_repr.make_images(frames, y_normals_resized).shape == (1, 64, 128, 3)

if __name__ == "__main__":
    test_dpt()
