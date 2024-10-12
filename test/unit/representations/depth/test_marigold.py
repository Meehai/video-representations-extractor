import numpy as np
from vre.representations.depth.marigold import Marigold
from vre.utils import FakeVideo

def test_marigold():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    marigold_repr = Marigold("testing", denoising_steps=1, ensemble_size=1, processing_resolution=30,
                             name="marigold", dependencies=[])
    marigold_repr.vre_setup(load_weights=False)
    marigold_repr.video = video

    frames = np.array(video[0:1])
    y_marigold = marigold_repr(frames)
    assert y_marigold.output.shape == (1, 8, 24), y_marigold.output.shape

    y_marigold_images = marigold_repr.make_images(frames, y_marigold)
    assert y_marigold_images.shape == (1, 8, 24, 3), y_marigold_images.shape
    assert y_marigold_images.dtype == np.uint8, y_marigold_images.dtype

    assert marigold_repr.size(y_marigold) == (8, 24)
    y_normals_resized = marigold_repr.resize(y_marigold, (64, 128)) # we can resize it though
    assert marigold_repr.size(y_normals_resized) == (64, 128)
    assert marigold_repr.make_images(frames, y_normals_resized).shape == (1, 64, 128, 3)

if __name__ == "__main__":
    test_marigold()
