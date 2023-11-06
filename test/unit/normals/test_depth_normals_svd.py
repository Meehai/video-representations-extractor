import numpy as np
from vre.representations.depth.odo_flow import DepthOdoFlow
from vre.representations.optical_flow.rife import FlowRife
from vre.representations.optical_flow.raft import FlowRaft
from vre.representations.normals.depth_svd import DepthNormalsSVD

class FakeVideo:
    def __init__(self, data: np.ndarray, frame_rate: int):
        self.data = data
        self.frame_rate = frame_rate

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, ix):
        return self.data[ix]

    def __len__(self):
        return len(self.data)

def test_depth_normals_svd_raft():
    rgb_data = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), 30)
    raft_repr = FlowRaft(video=rgb_data, name="raft", dependencies=[], inference_height=128, inference_width=128,
                         iters=2, small=True, mixed_precision=False)
    odoflow_repr = DepthOdoFlow(video=rgb_data, name="odoflow_rife", dependencies=[raft_repr],
                                linear_ang_vel_correction=True, focus_correction=True, cosine_correction_scipy=True,
                                cosine_correction_gd=False, sensor_fov=30, sensor_width=100, sensor_height=100,
                                min_depth_meters=0, max_depth_meters=100)
    depth_svd_normals_repr = DepthNormalsSVD(video=rgb_data, name="depth_svd_normals", dependencies=[odoflow_repr],
                                             sensor_fov=30, sensor_width=100, sensor_height=100, window_size=5,
                                             input_downsample_step=1, stride=1, max_distance=100, min_valid_count=0)

    y_depth_svd_normals, extra = depth_svd_normals_repr(slice(0, 1))
    assert y_depth_svd_normals.shape == (1, 128, 128, 3), y_depth_svd_normals.shape
    assert len(extra) == 0, extra

    y_depth_svd_normals_images = depth_svd_normals_repr.make_images(y_depth_svd_normals, extra)
    assert y_depth_svd_normals_images.shape == (1, 128, 128, 3), y_depth_svd_normals_images.shape
    assert y_depth_svd_normals_images.dtype == np.uint8, y_depth_svd_normals_images.dtype

def test_depth_normals_svd_rife():
    rgb_data = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), 30)
    rife_repr = FlowRife(video=rgb_data, name="rife", dependencies=[], compute_backward_flow=False, uhd=False)
    odoflow_repr = DepthOdoFlow(video=rgb_data, name="odoflow_rife", dependencies=[rife_repr],
                                linear_ang_vel_correction=True, focus_correction=True, cosine_correction_scipy=True,
                                cosine_correction_gd=False, sensor_fov=30, sensor_width=100, sensor_height=100,
                                min_depth_meters=0, max_depth_meters=100)
    depth_svd_normals_repr = DepthNormalsSVD(video=rgb_data, name="depth_svd_normals", dependencies=[odoflow_repr],
                                             sensor_fov=30, sensor_width=100, sensor_height=100, window_size=5,
                                             input_downsample_step=1, stride=1, max_distance=100, min_valid_count=0)

    y_depth_svd_normals, extra = depth_svd_normals_repr(slice(0, 1))
    assert y_depth_svd_normals.shape == (1, 64, 64, 3), y_depth_svd_normals.shape
    assert len(extra) == 0, extra

    y_depth_svd_normals_images = depth_svd_normals_repr.make_images(y_depth_svd_normals, extra)
    assert y_depth_svd_normals_images.shape == (1, 64, 64, 3), y_depth_svd_normals_images.shape
    assert y_depth_svd_normals_images.dtype == np.uint8, y_depth_svd_normals_images.dtype
