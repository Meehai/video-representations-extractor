from vre.utils import array_blend, str_topk
import numpy as np
import pytest

def test_image_blend_scalar():
    x = np.full(shape=(100, 100, 1), fill_value=0, dtype=np.float32)
    y = np.full(shape=(100, 100, 1), fill_value=1, dtype=np.float32)
    assert (array_blend(x, y, 0.5) == 0.5).all()

def test_image_blend_array():
    x = np.full(shape=(100, 100, 1), fill_value=0, dtype=np.float32)
    y = np.full(shape=(100, 100, 1), fill_value=1, dtype=np.float32)
    alpha = np.full(shape=x.shape, fill_value=0.3, dtype=np.float32)
    assert (array_blend(x, y, alpha) == 0.3).all()

def test_image_blend_bad_shapes():
    with pytest.raises(AssertionError):
        _ = array_blend(np.random.randn(100, 100), np.random.randn(50, 200), 0.5)
    with pytest.raises(AssertionError):
        _ = array_blend(np.random.randn(100, 100), np.random.randn(100, 100), np.random.randn(50, 200))
    with pytest.raises(AssertionError):
        _ = array_blend(np.random.randn(100, 100), np.random.randn(100, 100), 5)

def test_str_topk():
    assert str_topk("hello", 10) == "hello"
    assert str_topk("hello", 4) == "h..o"
    assert str_topk("helloworld", 4) == "h..d"
    assert str_topk("helloworld", 5) == "he..d"
    assert str_topk("helloworld", 6) == "he..ld"
    assert str_topk("helloworld", 7) == "hel..ld"
