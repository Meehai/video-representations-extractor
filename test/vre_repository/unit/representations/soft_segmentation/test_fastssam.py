import numpy as np
from vre_repository.soft_segmentation.fastsam import FastSam
from vre import FrameVideo

def test_fastsam():
    np.random.seed(42)
    video = FrameVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), 30)
    fastsam_repr = FastSam(variant="testing", iou=0.9, conf=0.4, name="fastsam", dependencies=[])
    fastsam_repr.vre_setup(load_weights=False)
    assert fastsam_repr.name == "fastsam"
    assert fastsam_repr.compress is True # default
    assert fastsam_repr.device == "cpu" # default from LearnedRepresentationMixin

    out = fastsam_repr.compute(video, [0])
    assert fastsam_repr.size(out) == (1, 64, 128, 3)
    assert out.output.shape == (1, 32, 128, 256)
    assert len(out.extra) == 1 # no objects in random images
    assert out.extra[0]["boxes"].shape == (300, 38)
    assert out.extra[0]["inference_size"] == (32, 64)
    assert out.extra[0]["image_size"] == (64, 128)

    out_image = fastsam_repr.make_images(out)
    assert out_image.shape == (1, 64, 128, 3) # no guarantee about the raw inference output
    assert out_image.dtype == np.uint8
    assert np.allclose(out_image[0], video[0])

    out_resized = fastsam_repr.resize(out, (80, 160)) # we can resize it though
    assert fastsam_repr.size(out_resized) == (1, 80, 160, 3)
    assert out_resized.output.shape == (1, 32, 128, 256)

if __name__ == "__main__":
    test_fastsam()
