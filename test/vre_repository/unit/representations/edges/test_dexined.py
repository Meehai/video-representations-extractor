import numpy as np
from vre_repository.edges.dexined import DexiNed
from vre import FakeVideo

def test_dexined_1():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), fps=30)
    dexined_repr = DexiNed(name="dexined", dependencies=[])
    dexined_repr.vre_setup(load_weights=False)
    assert dexined_repr.name == "dexined"
    assert dexined_repr.compress is True # default
    assert dexined_repr.device == "cpu" # default from LearnedRepresentationMixin

    out = dexined_repr.compute(video, [0])
    assert out.output.shape == (1, 512, 512, 1), out.output.shape

    out_images = dexined_repr.make_images(out)
    assert out_images.shape == (1, 512, 512, 3)
    assert out_images.dtype == np.uint8, out_images.dtype

    assert dexined_repr.size(out) == (1, 512, 512, 1)
    out_resized = dexined_repr.resize(out, (32, 64)) # we can resize it though
    assert dexined_repr.size(out_resized) == (1, 32, 64, 1)
    assert dexined_repr.make_images(out_resized).shape == (1, 32, 64, 3)

if __name__ == "__main__":
    test_dexined_1()
