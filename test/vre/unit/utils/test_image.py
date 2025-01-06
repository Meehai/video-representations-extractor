import pytest
from vre.utils.image import _get_rows_cols, collage_fn, image_blend, image_resize
import numpy as np

def test_get_rows_cols():
    assert _get_rows_cols(n_imgs=9, rows_cols=None) == (3, 3)
    assert _get_rows_cols(n_imgs=10, rows_cols=None) == (3, 4)
    assert _get_rows_cols(n_imgs=9, rows_cols=(2, 5)) == (2, 5)
    with pytest.raises(AssertionError):
        _get_rows_cols(n_imgs=9, rows_cols=(2, 4)) == (2, 5)
    assert _get_rows_cols(n_imgs=9, rows_cols=(2, -1)) == (2, 5)
    assert _get_rows_cols(n_imgs=9, rows_cols=(2, None)) == (2, 5)
    assert _get_rows_cols(n_imgs=9, rows_cols=(-1, 5)) == (2, 5)
    assert _get_rows_cols(n_imgs=9, rows_cols=(None, 5)) == (2, 5)
    assert _get_rows_cols(n_imgs=10, rows_cols=(None, 5)) == (2, 5)
    assert _get_rows_cols(n_imgs=11, rows_cols=(None, 5)) == (3, 5)
    assert _get_rows_cols(n_imgs=11, rows_cols=(None, 4)) == (3, 4)
    with pytest.raises(AssertionError):
        _ = _get_rows_cols(n_imgs=11, rows_cols=(None, None))
    with pytest.raises(AssertionError):
        _ = _get_rows_cols(n_imgs=None, rows_cols=None)
    with pytest.raises(AssertionError):
        _ = _get_rows_cols(n_imgs=0, rows_cols=None)
    with pytest.raises(AssertionError):
        _ = _get_rows_cols(n_imgs=-2, rows_cols=None)

def test_collage_fn_1():
    images = np.random.randint(0, 255, size=(3, 420, 420, 3), dtype=np.uint8)
    collage = collage_fn(images)
    assert collage.shape == (840, 840, 3)

def test_collage_fn_pad_1():
    images = np.random.randint(0, 255, size=(3, 420, 420, 3), dtype=np.uint8)
    collage = collage_fn(images, pad_bottom=20, pad_right=30, rows_cols=(2, 2))
    assert collage.shape == (860, 870, 3)

def test_collage_fn_pad_2():
    images = np.random.randint(0, 255, size=(3, 420, 420, 3), dtype=np.uint8)
    collage = collage_fn(images, pad_bottom=20, pad_right=30, rows_cols=(3, 1))
    assert collage.shape == (1300, 420, 3)

def test_collage_fn_pad_3():
    images = np.random.randint(0, 255, size=(3, 420, 420, 3), dtype=np.uint8)
    collage = collage_fn(images, pad_bottom=20, pad_right=30, rows_cols=(1, 3))
    assert collage.shape == (420, 1320, 3)

def test_collage_fn_none_images():
    images = [np.random.randint(0, 255, size=(420, 420, 3), dtype=np.uint8) for _ in range(3)]
    images = [images[0], None, images[1], None, images[2]]
    collage = collage_fn(images)
    assert collage.shape == (840, 1260, 3)

def test_collage_fn_pad_to_max():
    desired_shape = (420, 420, 3)
    images = [np.random.randint(0, 255, size=desired_shape, dtype=np.uint8)]
    for _ in range(8):
        shape = (desired_shape[0] - np.random.randint(10), desired_shape[1] - np.random.randint(10), 3)
        new_img = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        images.append(new_img)

    with pytest.raises(ValueError):
        collage_fn(images)

    collage = collage_fn(images, pad_to_max=True, rows_cols=(3, 3))
    assert collage.shape == (420 * 3, 420 * 3, 3)

def test_image_blend():
    with pytest.raises(AssertionError):
        image_blend(np.random.randn(100, 200, 3), np.random.randn(100, 200, 3), 0.5)

    x = np.full(shape=(100, 100, 1), fill_value=0, dtype=np.uint8)
    y = np.full(shape=(100, 100, 1), fill_value=100, dtype=np.uint8)
    assert (image_blend(x, y, 0.4) == 40).all()
    assert (image_blend(x, y, np.full((100, 100, 1), 0.6)) == 60).all()

def test_image_resize_width_None():
    assert image_resize(np.random.randint(0, 255, size=(100, 200, 3)), height=200, width=None).shape == (200, 400, 3)
