import numpy as np
from vre_repository.soft_segmentation.halftone import Halftone
from vre import FrameVideo

def test_halftone():
    video = FrameVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    halftone_repr = Halftone(name="halftone", dependencies=[], sample=3, scale=1, percentage=91,
                             angles=[0, 15, 30, 45], antialias=False, resolution=video.frame_shape[0:2])
    assert halftone_repr.name == "halftone"
    assert halftone_repr.compress is True # default from IORepresentationMixin

    out = halftone_repr.compute(video, [0])
    assert out.output.shape == (1, 64, 128, 3)
    out_images = halftone_repr.make_images(out)
    assert out_images.shape == (1, 64, 128, 3)
    assert out_images.dtype == np.uint8

    assert halftone_repr.size(out) == (1, 64, 128, 3)
    out_resized = halftone_repr.resize(out, (32, 64))
    assert halftone_repr.size(out_resized) == (1, 32, 64, 3)
    assert halftone_repr.make_images(out_resized).shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_halftone()
