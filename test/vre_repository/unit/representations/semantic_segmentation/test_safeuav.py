import numpy as np
from vre_repository.semantic_segmentation.safeuav import SafeUAV
from vre import FakeVideo

def test_safeuav():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    safeuav_repr = SafeUAV(name="safeuav", dependencies=[], variant="testing", disk_data_argmax=True)
    safeuav_repr.vre_setup(load_weights=False)
    assert safeuav_repr.name == "safeuav"
    assert safeuav_repr.compress is True # default
    assert safeuav_repr.device == "cpu" # default from LearnedRepresentationMixin

    out = safeuav_repr.compute(video, [0])
    assert out.output.shape == (1, 50, 50, 8)

    out_image = safeuav_repr.make_images(out)
    assert out_image.shape == (1, 50, 50, 3)
    assert out_image.dtype == np.uint8, out_image.dtype

    assert safeuav_repr.size(out) == (1, 50, 50, 8)
    out_resized = safeuav_repr.resize(out, (64, 128)) # we can resize it though
    assert safeuav_repr.size(out_resized) == (1, 64, 128, 8)
    assert safeuav_repr.make_images(out_resized).shape == (1, 64, 128, 3)

if __name__ == "__main__":
    test_safeuav()
