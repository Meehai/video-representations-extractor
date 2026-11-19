"""
SIFT (numpy, vre_repository.descriptor.sift) vs cv2.SIFT_create() with the same defaults (3 layers per octave,
sigma 1.6, contrast threshold 0.04, edge threshold 10) on the same grey image.
Not bit-exact: the float32 blur sums in another order and cv2 uses approximate atan2/exp, which moves a few orientations
by ~1 degree and flips secondary orientation peaks close to the 80% cutoff. So the checks are: >= 98% of the keypoints
match both ways (pt and size within 0.01, angle within 0.5 degrees) and matched keypoints have the same octave and
layer, the same response (rtol 1e-3) and descriptors within a few units of their 0..255 range.
Runs only if opencv is installed.
"""
import numpy as np
import pytest
from PIL import Image
from vre.utils import get_project_root
from vre_repository.descriptor.sift import computeKeypointsAndDescriptors

cv2 = pytest.importorskip("cv2")

@pytest.mark.parametrize("max_keypoints", [None, 300]) # cv2: nfeatures=0 (all) and nfeatures=300 (strongest 300)
@pytest.mark.parametrize("name", ["demo1.jpg", "demo2.jpg"])
def test_parity_sift_vs_cv2(name: str, max_keypoints: int | None):
    gray = np.array(Image.open(get_project_root() / "resources" / name).convert("L"))
    kps_cv2, desc_cv2 = cv2.SIFT_create(nfeatures=max_keypoints or 0).detectAndCompute(gray, None)
    kps, desc = computeKeypointsAndDescriptors(gray, max_keypoints=max_keypoints)

    pt_cv2, pt = np.array([kp.pt for kp in kps_cv2]), np.array([kp.pt for kp in kps], dtype=np.float64)
    size_cv2, size = np.array([kp.size for kp in kps_cv2]), np.array([kp.size for kp in kps], dtype=np.float64)
    angle_cv2, angle = np.array([kp.angle for kp in kps_cv2]), np.array([kp.angle for kp in kps], dtype=np.float64)
    same = ((np.hypot(pt_cv2[:, None, 0] - pt[None, :, 0], pt_cv2[:, None, 1] - pt[None, :, 1]) < 0.01) &
            (np.abs(size_cv2[:, None] - size[None]) < 0.01) &
            (np.abs((angle_cv2[:, None] - angle[None] + 180) % 360 - 180) < 0.5)) # (n_cv2, n): same keypoint
    assert same.any(axis=1).mean() >= 0.98, f"{same.any(axis=1).mean():.2%} of cv2's {len(kps_cv2)} kps found"
    assert same.any(axis=0).mean() >= 0.98, f"{same.any(axis=0).mean():.2%} of our {len(kps)} kps found by cv2"

    ix_cv2 = np.flatnonzero(same.any(axis=1))
    ix = same[ix_cv2].argmax(axis=1) # our keypoint matching each matched cv2 keypoint
    assert all((kps_cv2[i].octave & 0xFFFF) == (kps[j].octave & 0xFFFF) for i, j in zip(ix_cv2, ix)) # octave, layer
    assert np.allclose([kps_cv2[i].response for i in ix_cv2], [kps[j].response for j in ix], rtol=1e-3, atol=0)
    desc_diff = np.abs(desc_cv2[ix_cv2] - desc[ix])
    assert desc_diff.max() <= 8 and desc_diff.mean() <= 0.05, (desc_diff.max(), desc_diff.mean())
