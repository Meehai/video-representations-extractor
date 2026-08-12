import numpy as np
import pytest
from image_utils import image_draw_polygon, Color, image_draw_line, PointIJ
from image_utils_pil import image_draw_line_pil, image_draw_polygon_pil

np.set_printoptions(linewidth=200)
TOL = 0.0405 * 255

def white(*size) -> np.ndarray:
    return (np.ones((*size, 3)) * Color.WHITE).astype(np.uint8)

@pytest.mark.parametrize("size,p1,p2,thickness", [
    ((20 , 20 ), (10 , 10 ), (15 , 15 ), 1),
    ((20 , 20 ), (5  , 5  ), (15 , 15 ), 1),
    ((20 , 20 ), (2  , 7  ), (7  , 7  ), 20),
    ((250, 250), (100, 100), (100, 150), 20),
    ((10, 10), (2, 8), (7, 8), 20),
    ((30, 250), (10, 100), (10, 150), 20),
    ((15, 75), (5, 25), (5, 50), 50),
    ((10, 20), (5, 5), (5, 15), 1),
    ((20, 20), (10, 15), (10, 10), 10),
    ((10, 10), (2, 2), (6, 6), 10), # mozaic pattern 1: a single line without a skip_one
    ((10, 10), (3, 3), (7, 9), 10), # not a normal diagonal.
])
def test_image_draw_line_vs_pil(size: tuple[int, int], p1: PointIJ, p2: PointIJ, thickness: float):
    img = white(*size)
    res_pil = image_draw_line_pil(img, p1=p1, p2=p2, color=Color.BLACK, thickness=thickness)
    res = image_draw_line(img, p1=p1, p2=p2, color=Color.BLACK, thickness=thickness)

    assert np.allclose(res_pil, res)

@pytest.mark.parametrize("size,p1,p2,thickness", [
    ((20, 20), (10, 15), (15, 20), 20),
    ((15, 10), (5, 5), (10, 10), 20), # also not a normal diagonal.
    ((10, 10), (2, 2), (6, 6), 50), # mozaic pattern 3: 5 lines: odd ones have a skip_one (+/-1)
    ((10, 10), (3, 2), (6, 8), 10), # line slightly not aligned
])
def test_image_draw_line_vs_pil_tol(size: tuple[int, int], p1: PointIJ, p2: PointIJ, thickness: float):
    """close enough ones!"""
    img = white(*size)
    res_pil = image_draw_line_pil(img, p1=p1, p2=p2, color=Color.BLACK, thickness=thickness)
    res = image_draw_line(img, p1=p1, p2=p2, color=Color.BLACK, thickness=thickness)

    assert 0 < (res_pil - res).__abs__().mean().item() <= TOL

@pytest.mark.xfail
@pytest.mark.parametrize("size,p1,p2,thickness", [
    ((10, 10), (2, 2), (6, 6), 30), # mozaic pattern 2: 3 lines: middle one has a skip_one (+/-0)
    ((10, 10), (2, 2), (6, 6), 70), # mozaic pattern 4: 7 lines: odd ones have a skip_one (+/-1, +/-3)
    ((20, 20), (5, 5), (15, 15), 10), # big thickness causes mismatches
])
def test_image_draw_line_vs_pil_x(size: tuple[int, int], p1: PointIJ, p2: PointIJ, thickness: float):
    """xfail for image_draw_line. It's hard to replicate a curve with ifs. If you fix any of this, make a PR :)."""
    img = white(*size)
    res_pil = image_draw_line_pil(img, p1=p1, p2=p2, color=Color.BLACK, thickness=thickness)
    res = image_draw_line(img, p1=p1, p2=p2, color=Color.BLACK, thickness=thickness)
    assert 0 < (res_pil - res).__abs__().mean().item() <= TOL

@pytest.mark.parametrize("thickness", [1, 2, 3, 4, 10])
def test_image_draw_polygon_vs_pil_tol(thickness: float):
    img = white(20, 20)
    points = [(10, 10), (10, 15), (15, 15), (15, 20), (20, 10)]
    res_pil = image_draw_polygon_pil(img, points, color=Color.BLACK, thickness=thickness)
    res = image_draw_polygon(img, points, color=Color.BLACK, thickness=thickness)
    assert 0 < (res_pil - res).__abs__().mean().item() <= TOL

if __name__ == "__main__":
    test_image_draw_polygon_vs_pil_tol(10)
