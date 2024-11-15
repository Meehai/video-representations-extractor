import numpy as np
from vre.representations.soft_segmentation.fastsam import FastSam
from vre.utils import FakeVideo

def test_fastsam():
    np.random.seed(42)
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), 30)
    fastsam_repr = FastSam(variant="testing", iou=0.9, conf=0.4, name="fastsam", dependencies=[])
    fastsam_repr.vre_setup(load_weights=False)
    assert fastsam_repr.name == "fastsam"
    assert fastsam_repr.compress is True # default from ComputeRepresentationMixin
    assert fastsam_repr.device == "cpu" # default from LearnedRepresentationMixin

    fastsam_repr.compute(video, [0])
    assert fastsam_repr.size == (1, 64, 128, 3), fastsam_repr.size
    assert fastsam_repr.data.output.shape == (1, 32, 128, 256), fastsam_repr.data.output.shape
    assert len(fastsam_repr.data.extra) == 1 # no objects in random images
    assert fastsam_repr.data.extra[0]["boxes"].shape == (300, 38), fastsam_repr.data.extra[0]["boxes"].shape
    assert fastsam_repr.data.extra[0]["inference_size"] == (32, 64), fastsam_repr.data.extra[0]["inference_size"]
    assert fastsam_repr.data.extra[0]["image_size"] == (64, 128), fastsam_repr.data.extra[0]["image_size"]

    y_fastsam_rgb = fastsam_repr.make_images()
    assert y_fastsam_rgb.shape == (1, 64, 128, 3) # no guarantee about the raw inference output
    assert y_fastsam_rgb.dtype == np.uint8, y_fastsam_rgb.dtype
    assert fastsam_repr.size == (1, 64, 128, 3)
    assert np.allclose(y_fastsam_rgb[0], video[0])

    fastsam_repr.data = fastsam_repr.resize(fastsam_repr.data, (80, 160)) # we can resize it though
    assert fastsam_repr.size == (1, 80, 160, 3), fastsam_repr.size
    assert (A := fastsam_repr.data.output.shape) == (1, 32, 128, 256), A # this never changes

if __name__ == "__main__":
    test_fastsam()
