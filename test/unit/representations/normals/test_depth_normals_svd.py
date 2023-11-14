import numpy as np
from vre.representations.depth.dpt import DepthDpt
from vre.representations.depth.odo_flow import DepthOdoFlow
from vre.representations.optical_flow.rife import FlowRife
from vre.representations.optical_flow.raft import FlowRaft
from vre.representations.normals.depth_svd import DepthNormalsSVD
from vre.utils import FakeVideo

def test_depth_normals_svd_odoflow_raft():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    video_lin_vel = np.random.randn(len(video), 3)
    video_ang_vel = np.random.randn(len(video), 3)
    raft_repr = FlowRaft(name="raft", dependencies=[], inference_height=128, inference_width=128,
                         iters=2, small=True, mixed_precision=False)
    odoflow_repr = DepthOdoFlow(name="odoflow_rife", dependencies=[raft_repr],
                                linear_ang_vel_correction=True, focus_correction=True, cosine_correction_scipy=True,
                                cosine_correction_gd=False, sensor_fov=30, sensor_width=100, sensor_height=100,
                                min_depth_meters=0, max_depth_meters=100)
    depth_svd_normals_repr = DepthNormalsSVD(name="depth_svd_normals", dependencies=[odoflow_repr],
                                             sensor_fov=30, sensor_width=100, sensor_height=100, window_size=5,
                                             input_downsample_step=1, stride=1, max_distance=100, min_valid_count=0)

    frames = np.array(video[0:1])
    right_frames = np.array(video[1:2])
    lin_vel = video_lin_vel[1:2]
    ang_vel = video_ang_vel[1:2]

    y_raft = raft_repr(frames, right_frames=right_frames)
    y_odoflow, extra = odoflow_repr(frames, flows=y_raft, lin_vel=lin_vel, ang_vel=ang_vel)
    y_normals = depth_svd_normals_repr(frames, depths=y_odoflow)
    assert y_normals.shape == (1, y_raft.shape[1], y_raft.shape[2], 3), y_normals.shape

    y_normals_images = depth_svd_normals_repr.make_images(frames, y_normals)
    assert y_normals_images.shape == (1, 128, 128, 3), y_normals_images.shape
    assert y_normals_images.dtype == np.uint8, y_normals_images.dtype

def test_depth_normals_svd_odoflow_rife():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 64, 128, 3), dtype=np.uint8), frame_rate=30)
    video_lin_vel = np.random.randn(len(video), 3)
    video_ang_vel = np.random.randn(len(video), 3)
    rife_repr = FlowRife(name="rife", dependencies=[], compute_backward_flow=False, uhd=False)
    odoflow_repr = DepthOdoFlow(name="odoflow_rife", dependencies=[rife_repr],
                                linear_ang_vel_correction=True, focus_correction=True, cosine_correction_scipy=True,
                                cosine_correction_gd=False, sensor_fov=30, sensor_width=100, sensor_height=100,
                                min_depth_meters=0, max_depth_meters=100)
    depth_svd_normals_repr = DepthNormalsSVD(name="depth_svd_normals", dependencies=[odoflow_repr],
                                             sensor_fov=30, sensor_width=100, sensor_height=100, window_size=5,
                                             input_downsample_step=1, stride=1, max_distance=100, min_valid_count=0)

    frames = np.array(video[0:1])
    right_frames = np.array(video[1:2])
    lin_vel = video_lin_vel[1:2]
    ang_vel = video_ang_vel[1:2]

    y_rife = rife_repr(frames, right_frames=right_frames)
    y_odoflow, extra = odoflow_repr(frames, flows=y_rife, lin_vel=lin_vel, ang_vel=ang_vel)
    y_normals = depth_svd_normals_repr(frames, depths=y_odoflow)
    assert y_normals.shape == (1, y_rife.shape[1], y_rife.shape[2], 3), y_normals.shape

    y_normals_images = depth_svd_normals_repr.make_images(frames, y_normals)
    assert y_normals_images.shape == (1, 64, 128, 3), y_normals_images.shape
    assert y_normals_images.dtype == np.uint8, y_normals_images.dtype


def test_depth_normals_svd_dpt():
    video = FakeVideo(np.random.randint(0, 255, size=(20, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    dpt_repr = DepthDpt(name="dpt", dependencies=[])
    depth_svd_normals_repr = DepthNormalsSVD(name="depth_svd_normals_dpt", dependencies=[dpt_repr],
                                             sensor_fov=30, sensor_width=100, sensor_height=100, window_size=5,
                                             input_downsample_step=1, stride=1, max_distance=100, min_valid_count=0)

    frames = np.array(video[0:1])
    y_dpt = dpt_repr(frames)
    y_normals = depth_svd_normals_repr(frames, depths=y_dpt)
    assert y_normals.shape == (1, y_dpt.shape[1], y_dpt.shape[2], 3), y_normals.shape

    y_normals_img = depth_svd_normals_repr.make_images(frames, y_normals)
    assert y_normals_img.shape == (1, 128, 128, 3), y_normals_img.shape
