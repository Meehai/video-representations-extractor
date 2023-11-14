import numpy as np
from vre.representations.semantic_segmentation.fastsam import FastSam
from vre.utils import FakeVideo

def test_sseg_fastsam():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), 30)
    fastsam_repr = FastSam(variant="fastsam-s", iou=0.9, conf=0.4, name="fastsam-s", dependencies=[])

    frames = np.array(video[0:1])
    y_fastsam, extra = fastsam_repr(frames)
    assert y_fastsam.shape == (1, 32, 128, 256)
    # no objects in random images
    assert len(extra) == 1
    assert extra[0]["scaled_boxes"].shape == (0, 38)

    y_sseg_rgb = fastsam_repr.make_images(frames, (y_fastsam, extra))
    assert y_sseg_rgb.shape == (1, 64, 128, 3)
    assert y_sseg_rgb.dtype == np.uint8, y_sseg_rgb.dtype
    assert np.allclose(y_sseg_rgb, frames[0])

if __name__ == "__main__":
    test_sseg_fastsam()
