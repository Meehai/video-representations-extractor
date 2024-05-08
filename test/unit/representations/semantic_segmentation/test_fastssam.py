import numpy as np
from vre.representations.semantic_segmentation.fastsam import FastSam
from vre.utils import FakeVideo

def test_fastsam():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), 30)
    fastsam_repr = FastSam(variant="fastsam-s", iou=0.9, conf=0.4, name="fastsam-s", dependencies=[])

    frames = np.array(video[0:1])
    y_fastsam = fastsam_repr(frames)
    assert y_fastsam[0].shape == (1, 32, 128, 256)
    # no objects in random images
    assert len(y_fastsam[1]) == 1
    assert y_fastsam[1][0]["boxes"].shape == (0, 38) and y_fastsam[1][0]["inference_size"] == (512, 1024)

    y_fastsam_rgb = fastsam_repr.make_images(frames, y_fastsam)
    assert y_fastsam_rgb.shape == (1, 512, 1024, 3) # no guarantee about the raw inference output
    assert y_fastsam_rgb.dtype == np.uint8, y_fastsam_rgb.dtype
    assert fastsam_repr.size(y_fastsam) == (512, 1024)

    y_fastsam_resized = fastsam_repr.resize(y_fastsam, (64, 128)) # we can resize it though
    assert fastsam_repr.size(y_fastsam_resized) == (64, 128)
    y_fastsam_rgb_resized = fastsam_repr.make_images(frames, y_fastsam_resized)
    assert y_fastsam_rgb_resized.shape == (1, 64, 128, 3)
    assert np.allclose(y_fastsam_rgb_resized, frames[0])

if __name__ == "__main__":
    test_fastsam()
