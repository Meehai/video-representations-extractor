import numpy as np
from vre.representations.depth.odo_flow import DepthOdoFlow
from vre.representations.optical_flow.rife import FlowRife
from vre.representations.optical_flow.raft import FlowRaft
from vre.utils import FakeVideo

def test_odoflow_rife():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    video_lin_vel = np.random.randn(len(video), 3)
    video_ang_vel = np.random.randn(len(video), 3)

    rife_repr = FlowRife(name="rife", dependencies=[], compute_backward_flow=False, uhd=False)
    odoflow_repr = DepthOdoFlow(name="odoflow_rife", dependencies=[rife_repr],
                                linear_ang_vel_correction=True, focus_correction=True, sensor_fov=30,
                                sensor_width=100, sensor_height=100, min_depth_meters=0, max_depth_meters=100)

    frames = np.array(video[0:1])
    right_frames = np.array(video[1:2])
    lin_vel = video_lin_vel[1:2]
    ang_vel = video_ang_vel[1:2]
    y_rife = rife_repr(frames, right_frames=right_frames)
    y_odoflow = odoflow_repr(frames, flows=y_rife, lin_vel=lin_vel, ang_vel=ang_vel)
    assert y_odoflow[0].shape == (1, 32, 64), y_odoflow[0].shape

    y_odoflow_images = odoflow_repr.make_images(frames, y_odoflow)
    assert y_odoflow_images.shape == (1, 32, 64, 3), y_odoflow_images.shape
    assert y_odoflow_images.dtype == np.uint8, y_odoflow_images.dtype

    assert odoflow_repr.size(y_odoflow) == (32, 64)
    y_odoflow_resized = odoflow_repr.resize(y_odoflow, (64, 128)) # we can resize it though
    assert odoflow_repr.size(y_odoflow_resized) == (64, 128)
    assert odoflow_repr.make_images(frames, y_odoflow_resized).shape == (1, 64, 128, 3)

def test_odoflow_raft():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    video_lin_vel = np.random.randn(len(video), 3)
    video_ang_vel = np.random.randn(len(video), 3)

    raft_repr = FlowRaft(name="raft", dependencies=[], inference_height=128, inference_width=128, iters=2, small=True)
    odoflow_repr = DepthOdoFlow(name="odoflow_rife", dependencies=[raft_repr],
                                linear_ang_vel_correction=True, focus_correction=True, sensor_fov=30, sensor_width=100,
                                sensor_height=100, min_depth_meters=0, max_depth_meters=100)
    frames = np.array(video[0:1])
    right_frames = np.array(video[1:2])
    lin_vel = video_lin_vel[1:2]
    ang_vel = video_ang_vel[1:2]

    y_raft = raft_repr(frames, right_frames=right_frames)
    y_odoflow = odoflow_repr(frames, flows=y_raft, lin_vel=lin_vel, ang_vel=ang_vel)
    assert y_odoflow[0].shape == (1, 128, 128), y_odoflow[0].shape
    assert len(y_odoflow[1]) == 1, y_odoflow[1]

    y_odoflow_images = odoflow_repr.make_images(frames, y_odoflow)
    assert y_odoflow_images.shape == (1, 128, 128, 3), y_odoflow_images.shape
    assert y_odoflow_images.dtype == np.uint8, y_odoflow_images.dtype

    assert odoflow_repr.size(y_odoflow) == (128, 128)
    y_odoflow_resized = odoflow_repr.resize(y_odoflow, (64, 128)) # we can resize it though
    assert odoflow_repr.size(y_odoflow_resized) == (64, 128)
    assert odoflow_repr.make_images(frames, y_odoflow_resized).shape == (1, 64, 128, 3)
