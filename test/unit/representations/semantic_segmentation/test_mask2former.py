import numpy as np
from vre.representations.semantic_segmentation.mask2former import Mask2Former
from vre.utils import FakeVideo

def test_mask2former():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), 30)
    m2f_repr = Mask2Former(model_id="49189528_1", semantic_argmax_only=True, name="m2f", dependencies=[])
    m2f_repr.vre_setup(load_weights=False)
    assert m2f_repr.name == "m2f"
    assert m2f_repr.compress is True # default from ComputeRepresentationMixin
    assert m2f_repr.device == "cpu" # default from LearnedRepresentationMixin

    m2f_repr.compute(video, [0])
    assert m2f_repr.data.output.shape == (1, 64, 128)

    y_m2f_rgb = m2f_repr.make_images()
    assert y_m2f_rgb.shape == (1, 64, 128, 3)
    assert y_m2f_rgb.dtype == np.uint8, y_m2f_rgb.dtype

    assert m2f_repr.size == (1, 64, 128)
    m2f_repr.data = m2f_repr.resize(m2f_repr.data, (32, 64)) # we can resize it though
    assert m2f_repr.size == (1, 32, 64)
    assert m2f_repr.make_images().shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_mask2former()
