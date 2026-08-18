import pytest
import numpy as np
from image_utils import image_resize

np.set_printoptions(linewidth=200)

def gray(vals, dtype) -> np.ndarray:
    return np.stack([np.array(vals, dtype=dtype)] * 3, axis=-1)

@pytest.mark.parametrize("dtype, channels", [["uint8", 1], ["uint8", 3], ["float32", 1], ["float32", 3]])
def test_image_resize_maintain_channels(dtype: str, channels: int):
    image = (np.random.random(size=(100, 200, channels)) * 255).astype(dtype=dtype)
    assert image_resize(image, 50, 100).shape == (50, 100, channels)

@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_image_resize_width_None(dtype: str):
    image = (np.random.random(size=(100, 200, 3)) * 255).astype( dtype=dtype)
    assert image_resize(image, height=200, width=None).shape == (200, 400, 3)

@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_image_resize_upscale_nearest(dtype: str):
    """upscale 2x3 -> 4x6 with nearest interpolation"""
    image = [[  0, 128, 255],
             [255,  64,   0]]
    res = image_resize(gray(image, dtype), height=4, width=6, interpolation="nearest")
    expected = [[  0,   0, 128, 128, 255, 255],
                [  0,   0, 128, 128, 255, 255],
                [255, 255,  64,  64,   0,   0],
                [255, 255,  64,  64,   0,   0]]
    assert res.dtype == np.dtype(dtype), f"{res.dtype=}"
    assert (res.transpose(2, 0, 1) == expected).all() # no need to round

@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_image_resize_upscale_bilinear(dtype: str):
    """upscale 2x3 -> 4x6 with bilinear interpolation"""
    image = [[  0, 128, 255],
             [255,  64,   0]]
    res = image_resize(gray(image, dtype), height=4, width=6, interpolation="bilinear")
    expected = [[  0,  32,  96, 160, 223, 255],
                [ 64,  76, 100, 132, 171, 191],
                [191, 163, 108,  76,  68,  64],
                [255, 207, 112,  48,  16,   0]]
    assert res.dtype == np.dtype(dtype), f"{res.dtype=}"
    assert (res.transpose(2, 0, 1).round() == expected).all()

@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_image_resize_downscale_nearest(dtype: str):
    """downscale 6x6 -> 2x2 with nearest interpolation"""
    image = [[  0,   0,   0, 255, 255, 255],
             [  0,   0,   0, 255, 255, 255],
             [  0,   0,   0, 255, 255, 255],
             [128, 128, 128,  64,  64,  64],
             [128, 128, 128,  64,  64,  64],
             [128, 128, 128,  64,  64,  64]]
    res = image_resize(gray(image, dtype), height=2, width=2, interpolation="nearest")
    expected = [[  0, 255],
                [128,  64]]
    assert res.dtype == np.dtype(dtype), f"{res.dtype=}"
    assert (res.transpose(2, 0, 1) == expected).all() # no need to round

@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_image_resize_downscale_bilinear(dtype: str):
    """downscale 6x6 -> 2x2 with bilinear interpolation"""
    image = [[  0,   0,   0, 255, 255, 255],
             [  0,   0,   0, 255, 255, 255],
             [  0,   0,   0, 255, 255, 255],
             [128, 128, 128,  64,  64,  64],
             [128, 128, 128,  64,  64,  64],
             [128, 128, 128,  64,  64,  64]]
    res = image_resize(gray(image, dtype), height=2, width=2, interpolation="bilinear", backend="PIL")
    expected = [[ 43, 204],
                [109,  91]]
    assert res.dtype == np.dtype(dtype), f"{res.dtype=}"
    assert (res.transpose(2, 0, 1).round() == expected).all()
