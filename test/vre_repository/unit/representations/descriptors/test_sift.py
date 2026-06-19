import numpy as np
from vre_repository.descriptors.sift import SIFT
from vre_video import VREVideo

def test_sift():
    video = VREVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    sift_repr = SIFT(name="sift", dependencies=[])
    assert sift_repr.name == "sift"
    assert sift_repr.compress is True # default

    out = sift_repr.compute(video, [0])
    breakpoint()
    assert out.output.shape == (1, 50, 50, 8)

    out_image = sift_repr.make_images(out)
    assert out_image.shape == (1, 50, 50, 3)
    assert out_image.dtype == np.uint8, out_image.dtype

    assert sift_repr.size(out) == (1, 50, 50, 8)
    out_resized = sift_repr.resize(out, (64, 128)) # we can resize it though
    assert sift_repr.size(out_resized) == (1, 64, 128, 8)
    assert sift_repr.make_images(out_resized).shape == (1, 64, 128, 3)

if __name__ == "__main__":
    test_sift()
