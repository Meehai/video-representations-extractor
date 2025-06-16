import numpy as np
from vre_repository.depth.dpt import DepthDpt
from vre import FrameVideo

def test_dpt():
    video = FrameVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    dpt_repr = DepthDpt(name="dpt", dependencies=[])
    dpt_repr.vre_setup(load_weights=False)
    assert dpt_repr.name == "dpt"
    assert dpt_repr.compress is True # default
    assert dpt_repr.device == "cpu" # default from LearnedRepresentationMixin

    out = dpt_repr.compute(video, ixs=[0])
    assert out.output.shape == (1, 192, 384, 1), out.output.shape

    out_images = dpt_repr.make_images(out)
    assert out_images.shape == (1, 192, 384, 3), out_images.shape
    assert out_images.dtype == np.uint8, out_images.dtype

    assert dpt_repr.size(out) == (1, 192, 384, 1)
    out_resized = dpt_repr.resize(out, (64, 128)) # we can resize it though
    assert dpt_repr.size(out_resized) == (1, 64, 128, 1)
    assert dpt_repr.make_images(out_resized).shape == (1, 64, 128, 3)

if __name__ == "__main__":
    test_dpt()
