import numpy as np
from vre.representations.depth.dpt import DepthDpt
from vre.representations.normals.depth_svd import DepthNormalsSVD
from vre.utils import FakeVideo

def test_depth_normals_svd_dpt():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 128, 128, 3), dtype=np.uint8), fps=30)
    dpt_repr = DepthDpt(name="dpt", dependencies=[])
    normals_svd_repr = DepthNormalsSVD(name="depth_svd_normals_dpt", dependencies=[dpt_repr],
                                       sensor_fov=30, sensor_width=100, sensor_height=100, window_size=5,
                                       input_downsample_step=1, stride=1, max_distance=100, min_valid_count=0)
    dpt_repr.vre_setup(load_weights=False)
    assert normals_svd_repr.name == "depth_svd_normals_dpt"
    assert normals_svd_repr.compress is True # default from ComputeRepresentationMixin

    dpt_repr.compute(video, ixs=[0])
    normals_svd_repr.compute(video, ixs=[0])
    assert normals_svd_repr.data.output.shape == (*dpt_repr.data.output.shape, 3)

    y_normals_img = normals_svd_repr.make_images()
    assert y_normals_img.shape == (1, 384, 384, 3), y_normals_img.shape

    assert normals_svd_repr.size == (1, 384, 384, 3)
    normals_svd_repr.data = normals_svd_repr.resize(normals_svd_repr.data, (64, 128)) # we can resize it though
    assert normals_svd_repr.size == (1, 64, 128, 3)
    assert normals_svd_repr.make_images().shape == (1, 64, 128, 3)
