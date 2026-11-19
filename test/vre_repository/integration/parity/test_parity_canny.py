"""
Canny edges (numpy, vre_repository.edges.canny) vs cv2.Canny: same image and parameters, same edge pixels. Bit-exact:
the Sobel gradients are integer sums and every threshold, direction and tie test is on integers, so any mismatch is a
real behaviour difference. Runs only if opencv is installed.
"""
import numpy as np
from PIL import Image
import pytest
from vre_video import VREVideo
from vre.utils import get_project_root
from vre_repository.edges.canny import Canny, canny

cv2 = pytest.importorskip("cv2")

@pytest.mark.parametrize("aperture_size, scale", [(3, 1), (5, 12), (7, 160)]) # scale: thresholds to magnitude range
@pytest.mark.parametrize("l2_gradient", [False, True])
def test_parity_canny_random_vs_cv2(aperture_size: int, scale: int, l2_gradient: bool):
    """noise, blocks and 2-3 grey levels (exact magnitude ties); 1x1 to 79x79; grey, RGB and RGBA; any thresholds"""
    rng = np.random.default_rng(42)
    for _ in range(300):
        h, w, block = (int(x) for x in rng.integers(1, [80, 80, 6]))
        c, levels = int(rng.choice([1, 3, 4])), int(rng.choice([2, 3, 256]))
        image = rng.integers(0, levels, size=(h // block + 1, w // block + 1, c)).repeat(block, 0).repeat(block, 1)
        image = (image[0:h, 0:w] * (255 // (levels - 1))).astype(np.uint8)
        image = image[..., 0] if c == 1 else image
        t1, t2 = rng.uniform(-20, 600, size=2) * scale # unordered, fractional and negative thresholds
        expected = cv2.Canny(image, t1, t2, apertureSize=aperture_size, L2gradient=l2_gradient) == 255
        assert np.array_equal(canny(image, t1, t2, aperture_size, l2_gradient), expected), (h, w, c, block, t1, t2)

@pytest.mark.parametrize("aperture_size, threshold1, threshold2", [(3, 100, 200), (3, 10, 30), (5, 1000, 3000),
                                                                   (7, 16000, 48000)])
@pytest.mark.parametrize("l2_gradient", [False, True])
@pytest.mark.parametrize("name", ["demo1.jpg", "demo2.jpg"])
def test_parity_canny_photos_vs_cv2(name: str, aperture_size: int, threshold1: float, threshold2: float,
                                    l2_gradient: bool):
    """full-size photos, RGB and grey"""
    rgb = np.array(Image.open(get_project_root() / "resources" / name))
    for image in [rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)]:
        expected = cv2.Canny(image, threshold1, threshold2, apertureSize=aperture_size, L2gradient=l2_gradient) == 255
        assert np.array_equal(canny(image, threshold1, threshold2, aperture_size, l2_gradient), expected), image.shape

def test_parity_canny_representation_vs_cv2():
    """the representation's output is cv2.Canny's edge map as (1, H, W, 1) float32 in {0, 1}"""
    frame = np.array(Image.open(get_project_root() / "resources/demo1.jpg"))
    canny_repr = Canny(name="canny", dependencies=[], threshold1=100, threshold2=200, aperture_size=3, l2_gradient=True)
    y = canny_repr.compute(VREVideo(frame[None], fps=1), [0]).output
    expected = np.float32(cv2.Canny(frame, 100, 200, apertureSize=3, L2gradient=True))[None, ..., None] / 255
    assert y.dtype == np.float32 and np.array_equal(y, expected), (y.dtype, y.shape)
