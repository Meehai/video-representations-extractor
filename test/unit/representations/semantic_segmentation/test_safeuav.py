import numpy as np
from vre.representations.semantic_segmentation.safeuav import SafeUAV
from vre.utils import FakeVideo

def test_safeuav():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    train_height, train_width = 100, 200
    safeuav_repr = SafeUAV(name="safeuav", dependencies=[], num_classes=3, train_height=train_height,
                           train_width=train_width, color_map=[[0, 255, 0], [0, 127, 0], [255, 255, 0]])
    safeuav_repr.vre_setup(load_weights=False)

    frames = np.array(video[0:1])
    y_safeuav = safeuav_repr(frames)
    assert y_safeuav.output.shape == (1, train_height, train_width) # no guarantee that make() produces (vid_h, vid_w)

    y_safeuav_rgb = safeuav_repr.make_images(frames, y_safeuav)
    assert y_safeuav_rgb.shape == (1, train_height, train_width, 3)
    assert y_safeuav_rgb.dtype == np.uint8, y_safeuav_rgb.dtype

    assert safeuav_repr.size(y_safeuav) == (train_height, train_width)
    y_safeuav_resized = safeuav_repr.resize(y_safeuav, (64, 128)) # we can resize it though
    assert safeuav_repr.size(y_safeuav_resized) == (64, 128)
    assert safeuav_repr.make_images(frames, y_safeuav_resized).shape == (1, 64, 128, 3)

if __name__ == "__main__":
    test_safeuav()
