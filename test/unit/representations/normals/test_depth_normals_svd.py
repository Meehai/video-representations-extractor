import numpy as np
from vre.representations.depth.dpt import DepthDpt
from vre.representations.normals.depth_svd import DepthNormalsSVD
from vre.utils import FakeVideo

def test_depth_normals_svd_dpt():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    dpt_repr = DepthDpt(name="dpt", dependencies=[])
    depth_svd_normals_repr = DepthNormalsSVD(name="depth_svd_normals_dpt", dependencies=[dpt_repr],
                                             sensor_fov=30, sensor_width=100, sensor_height=100, window_size=5,
                                             input_downsample_step=1, stride=1, max_distance=100, min_valid_count=0)

    depth_svd_normals_repr.video = video
    frames = np.array(video[0:1])
    y_deps = depth_svd_normals_repr.vre_dep_data(slice(0, 1))
    # y_dpt = dpt_repr(frames)
    y_normals = depth_svd_normals_repr(frames, **y_deps)
    assert y_normals.shape == (1, y_deps["depths"].shape[1], y_deps["depths"].shape[2], 3), y_normals.shape

    y_normals_img = depth_svd_normals_repr.make_images(frames, y_normals)
    assert y_normals_img.shape == (1, 384, 384, 3), y_normals_img.shape

    assert depth_svd_normals_repr.size(y_normals) == (384, 384)
    y_normals_resized = depth_svd_normals_repr.resize(y_normals, (64, 128)) # we can resize it though
    assert depth_svd_normals_repr.size(y_normals_resized) == (64, 128)
    assert depth_svd_normals_repr.make_images(frames, y_normals_resized).shape == (1, 64, 128, 3)

