import numpy as np
from vre.representations.edges.dexined import DexiNed
from vre.utils import FakeVideo

def test_dexined_1():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    dexined_repr = DexiNed(name="dexined", dependencies=[], inference_height=512, inference_width=512)

    frames = np.array(video[0:1])
    y_dexined = dexined_repr(frames)
    assert y_dexined.shape == (1, 512, 512)

    y_dexined_images = dexined_repr.make_images(frames, y_dexined)
    assert y_dexined_images.shape == (1, 64, 128, 3)
    assert y_dexined_images.dtype == np.uint8, y_dexined_images.dtype

if __name__ == "__main__":
    test_dexined_1()
