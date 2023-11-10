import numpy as np
from vre.representations.depth.odo_flow import DepthOdoFlow
from vre.representations.optical_flow.rife import FlowRife
from vre.representations.optical_flow.raft import FlowRaft
from vre.utils import FakeVideo

def test_odoflow_rife():
    rgb_data = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), 30)
    rife_repr = FlowRife(video=rgb_data, name="rife", dependencies=[], compute_backward_flow=False, uhd=False)
    odoflow_repr = DepthOdoFlow(video=rgb_data, name="odoflow_rife", dependencies=[rife_repr],
                                linear_ang_vel_correction=True, focus_correction=True, cosine_correction_scipy=True,
                                cosine_correction_gd=False, sensor_fov=30, sensor_width=100, sensor_height=100,
                                min_depth_meters=0, max_depth_meters=100)

    y_odoflow, extra = odoflow_repr(slice(0, 1))
    assert y_odoflow.shape == (1, 64, 64), y_odoflow.shape
    assert len(extra) == 1, extra

    y_odoflow_images = odoflow_repr.make_images(slice(0, 1), y_odoflow, extra)
    assert y_odoflow_images.shape == (1, 128, 128, 3), y_odoflow_images.shape
    assert y_odoflow_images.dtype == np.uint8, y_odoflow_images.dtype

def test_odoflow_raft():
    rgb_data = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), 30)
    raft_repr = FlowRaft(video=rgb_data, name="raft", dependencies=[], inference_height=128, inference_width=128,
                         iters=2, small=True, mixed_precision=False)
    odoflow_repr = DepthOdoFlow(video=rgb_data, name="odoflow_rife", dependencies=[raft_repr],
                                linear_ang_vel_correction=True, focus_correction=True, cosine_correction_scipy=True,
                                cosine_correction_gd=False, sensor_fov=30, sensor_width=100, sensor_height=100,
                                min_depth_meters=0, max_depth_meters=100)

    y_odoflow, extra = odoflow_repr(slice(0, 1))
    assert y_odoflow.shape == (1, 128, 128), y_odoflow.shape
    assert len(extra) == 1, extra

    y_odoflow_images = odoflow_repr.make_images(slice(0, 1), y_odoflow, extra)
    assert y_odoflow_images.shape == (1, 128, 128, 3), y_odoflow_images.shape
    assert y_odoflow_images.dtype == np.uint8, y_odoflow_images.dtype
