import numpy as np
from vre_repository.semantic_segmentation.mask2former import Mask2Former
from vre import FakeVideo

def test_mask2former():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), 30)
    m2f_repr = Mask2Former(model_id="49189528_1", disk_data_argmax=True, name="m2f", dependencies=[])
    m2f_repr.vre_setup(load_weights=False)
    assert m2f_repr.name == "m2f"
    assert m2f_repr.compress is True # default
    assert m2f_repr.device == "cpu" # default from LearnedRepresentationMixin

    out = m2f_repr.compute(video, [0])
    assert out.output.shape == (1, 64, 128, m2f_repr.n_classes)

    out_image = m2f_repr.make_images(out)
    assert out_image.shape == (1, 64, 128, 3)
    assert out_image.dtype == np.uint8, out_image.dtype

    assert m2f_repr.size(out) == (1, 64, 128, m2f_repr.n_classes)
    out_resized = m2f_repr.resize(out, (32, 64)) # we can resize it though
    assert m2f_repr.size(out_resized) == (1, 32, 64, m2f_repr.n_classes)
    assert m2f_repr.make_images(out_resized).shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_mask2former()
