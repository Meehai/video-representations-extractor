import numpy as np
from vre.representations.depth.dpt import DepthDpt
from vre.utils import FakeVideo

def test_dpt():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    dpt_repr = DepthDpt(name="dpt", dependencies=[])
    dpt_repr.vre_setup(load_weights=False)
    assert dpt_repr.name == "dpt"
    assert dpt_repr.compress is True # default from ComputeRepresentationMixin
    assert dpt_repr.device == "cpu" # default from LearnedRepresentationMixin

    dpt_repr.compute(video, ixs=[0])
    assert dpt_repr.data.output.shape == (1, 192, 384), dpt_repr.data.output.shape

    y_dpt_images = dpt_repr.make_images()
    assert y_dpt_images.shape == (1, 192, 384, 3), y_dpt_images.shape
    assert y_dpt_images.dtype == np.uint8, y_dpt_images.dtype

    assert dpt_repr.size == (1, 192, 384)
    dpt_repr.data = dpt_repr.resize(dpt_repr.data, (64, 128)) # we can resize it though
    assert dpt_repr.size == (1, 64, 128)
    assert dpt_repr.make_images().shape == (1, 64, 128, 3)

if __name__ == "__main__":
    test_dpt()
