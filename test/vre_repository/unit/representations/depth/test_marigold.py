import numpy as np
from vre_repository.depth.marigold import Marigold
from vre import FakeVideo

def test_marigold():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    marigold_repr = Marigold("testing", denoising_steps=1, ensemble_size=1, processing_resolution=30,
                             name="marigold", dependencies=[])
    marigold_repr.vre_setup(load_weights=False)
    assert marigold_repr.name == "marigold"
    assert marigold_repr.compress is True # default from ComputeRepresentationMixin
    assert marigold_repr.device == "cpu" # default from LearnedRepresentationMixin

    out = marigold_repr.compute(video, ixs=[0])
    assert out.output.shape == (1, 8, 24, 1)

    out_images = marigold_repr.make_images(out)
    assert out_images.shape == (1, 8, 24, 3)
    assert out_images.dtype == np.uint8

    assert marigold_repr.size(out) == (1, 8, 24, 1)
    out_resized = marigold_repr.resize(out, (64, 128)) # we can resize it though
    assert marigold_repr.size(out_resized) == (1, 64, 128, 1)
    assert marigold_repr.make_images(out_resized).shape == (1, 64, 128, 3)

if __name__ == "__main__":
    test_marigold()
