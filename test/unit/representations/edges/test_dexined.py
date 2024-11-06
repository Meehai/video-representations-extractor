import numpy as np
from vre.representations.edges.dexined import DexiNed
from vre.utils import FakeVideo

def test_dexined_1():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    dexined_repr = DexiNed(name="dexined", dependencies=[])
    dexined_repr.vre_setup(load_weights=False)
    assert dexined_repr.name == "dexined"
    assert dexined_repr.compress is True # default from ComputeRepresentationMixin
    assert dexined_repr.device == "cpu" # default from LearnedRepresentationMixin

    frames = np.array(video[0:1])
    y_dexined = dexined_repr(frames)
    assert y_dexined.output.shape == (1, 512, 512), y_dexined.output

    y_dexined_images = dexined_repr.make_images(frames, y_dexined)
    assert y_dexined_images.shape == (1, 512, 512, 3)
    assert y_dexined_images.dtype == np.uint8, y_dexined_images.dtype

    assert dexined_repr.size(y_dexined) == (512, 512)
    y_dexined_resized = dexined_repr.resize(y_dexined, (32, 64)) # we can resize it though
    assert dexined_repr.size(y_dexined_resized) == (32, 64)
    assert dexined_repr.make_images(frames, y_dexined_resized).shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_dexined_1()
