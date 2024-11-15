import numpy as np
from vre.representations.edges.dexined import DexiNed
from vre.utils import FakeVideo

def test_dexined_1():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    dexined_repr = DexiNed(name="dexined", dependencies=[])
    dexined_repr.vre_setup(load_weights=False)
    assert dexined_repr.name == "dexined"
    assert dexined_repr.compress is True # default from ComputeRepresentationMixin
    assert dexined_repr.device == "cpu" # default from LearnedRepresentationMixin

    dexined_repr.compute(video, [0])
    assert dexined_repr.data.output.shape == (1, 512, 512), dexined_repr.data.output.shape

    y_dexined_images = dexined_repr.make_images()
    assert y_dexined_images.shape == (1, 512, 512, 3)
    assert y_dexined_images.dtype == np.uint8, y_dexined_images.dtype

    assert dexined_repr.size == (1, 512, 512)
    dexined_repr.data = dexined_repr.resize(dexined_repr.data, (32, 64)) # we can resize it though
    assert dexined_repr.size == (1, 32, 64)
    assert dexined_repr.make_images().shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_dexined_1()
