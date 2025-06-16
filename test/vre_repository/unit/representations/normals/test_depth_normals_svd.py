import numpy as np
from vre_repository.depth.dpt import DepthDpt
from vre_repository.normals.depth_svd import DepthNormalsSVD
from vre import FrameVideo

def test_depth_normals_svd_dpt():
    video = FrameVideo(np.random.randint(0, 255, size=(20, 128, 128, 3), dtype=np.uint8), fps=30)
    dpt_repr = DepthDpt(name="dpt", dependencies=[])
    normals_svd_repr = DepthNormalsSVD(name="depth_svd_normals_dpt", dependencies=[dpt_repr], sensor_fov=30,
                                       sensor_size=(100, 100), window_size=5, input_downsample_step=1, stride=1)
    dpt_repr.vre_setup(load_weights=False)
    assert normals_svd_repr.name == "depth_svd_normals_dpt"
    assert normals_svd_repr.compress is True # default

    out_dpt = dpt_repr.compute(video, ixs=[0])
    out = normals_svd_repr.compute(video, ixs=[0], dep_data=[out_dpt])
    assert out.output.shape == (*out_dpt.output.shape[0:-1], 3)

    out_images = normals_svd_repr.make_images(out)
    assert out_images.shape == (1, 384, 384, 3), out_images

    assert normals_svd_repr.size(out) == (1, 384, 384, 3)
    out_resized = normals_svd_repr.resize(out, (64, 128)) # we can resize it though
    assert normals_svd_repr.size(out_resized) == (1, 64, 128, 3)
    assert normals_svd_repr.make_images(out_resized).shape == (1, 64, 128, 3)
