import numpy as np
from vre.representations.semantic_segmentation.safeuav import SafeUAV
from vre.utils import FakeVideo

def test_sseg_safeuav():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    sseg_repr = SafeUAV(name="safeuav", dependencies=[], num_classes=3, train_height=100, train_width=200,
                        color_map=[[0, 255, 0], [0, 127, 0], [255, 255, 0]])

    frames = np.array(video[0:1])
    y_sseg = sseg_repr(frames)
    assert y_sseg.shape == (1, 100, 200)

    y_sseg_rgb = sseg_repr.make_images(frames, y_sseg)
    assert y_sseg_rgb.shape == (1, 64, 128, 3)
    assert y_sseg_rgb.dtype == np.uint8, y_sseg_rgb.dtype

if __name__ == "__main__":
    test_sseg_safeuav()
