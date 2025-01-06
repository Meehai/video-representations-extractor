import numpy as np
from vre_repository.semantic_segmentation.safeuav import SafeUAV
from vre.utils import FakeVideo

def test_safeuav():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    train_height, train_width = 100, 200
    num_classes = 7
    color_map = [[i, i, i] for i in range(num_classes)]
    safeuav_repr = SafeUAV(name="safeuav", dependencies=[], num_classes=7, train_height=train_height,
                           train_width=train_width, color_map=color_map, disk_data_argmax=True)
    safeuav_repr.vre_setup(load_weights=False)
    assert safeuav_repr.name == "safeuav"
    assert safeuav_repr.compress is True # default from ComputeRepresentationMixin
    assert safeuav_repr.device == "cpu" # default from LearnedRepresentationMixin

    safeuav_repr.compute(video, [0])
    assert safeuav_repr.data.output.shape == (1, train_height, train_width, 7)

    y_safeuav_rgb = safeuav_repr.make_images(safeuav_repr.data)
    assert y_safeuav_rgb.shape == (1, train_height, train_width, 3)
    assert y_safeuav_rgb.dtype == np.uint8, y_safeuav_rgb.dtype

    assert safeuav_repr.size == (1, train_height, train_width, 7)
    safeuav_repr.data = safeuav_repr.resize(safeuav_repr.data, (64, 128)) # we can resize it though
    assert safeuav_repr.size == (1, 64, 128, 7)
    assert safeuav_repr.make_images(safeuav_repr.data).shape == (1, 64, 128, 3)

if __name__ == "__main__":
    test_safeuav()
