import numpy as np
from vre.representations.edges.dexined import DexiNed

def test_dexined_1():
    rgb_data = np.random.randint(0, 255, size=(1, 64, 128, 3), dtype=np.uint8)
    dexined_repr = DexiNed(video=rgb_data, name="dexined", dependencies=[], inference_height=512, inference_width=512)
    y_dexined, extra = dexined_repr(slice(0, 1))
    assert y_dexined.shape == (1, 512, 512)
    assert extra == {}
    y_dexined_images = dexined_repr.make_images(y_dexined, extra)
    assert y_dexined_images.shape == (1, 512, 512, 3)
    assert y_dexined_images.dtype == np.uint8, y_dexined_images.dtype

if __name__ == "__main__":
    test_dexined_1()
