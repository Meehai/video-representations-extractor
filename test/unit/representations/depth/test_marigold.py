import numpy as np
from vre.representations.depth.marigold import Marigold
from vre.utils import FakeVideo

def test_marigold():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    marigold_repr = Marigold("testing", denoising_steps=1, ensemble_size=1, processing_resolution=30,
                             name="marigold", dependencies=[])
    marigold_repr.vre_setup(load_weights=False)
    assert marigold_repr.name == "marigold"
    assert marigold_repr.compress is True # default from ComputeRepresentationMixin
    assert marigold_repr.device == "cpu" # default from LearnedRepresentationMixin

    marigold_repr.compute(video, ixs=[0])
    assert marigold_repr.data.output.shape == (1, 8, 24), marigold_repr.data.output.shape

    y_dpt_images = marigold_repr.make_images()
    assert y_dpt_images.shape == (1, 8, 24, 3), y_dpt_images.shape
    assert y_dpt_images.dtype == np.uint8, y_dpt_images.dtype

    assert marigold_repr.size == (1, 8, 24)
    marigold_repr.data = marigold_repr.resize(marigold_repr.data, (64, 128)) # we can resize it though
    assert marigold_repr.size == (1, 64, 128)
    assert marigold_repr.make_images().shape == (1, 64, 128, 3)

if __name__ == "__main__":
    test_marigold()
