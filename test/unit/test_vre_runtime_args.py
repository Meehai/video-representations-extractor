import sys
from vre.utils import FakeVideo, get_project_root
from vre.vre_runtime_args import VRERuntimeArgs
import numpy as np
import pytest

sys.path.append(str(get_project_root() / "test"))
from fake_representation import FakeRepresentation

def test_VRERuntimeArgs_ctor():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    representations = {"rgb": FakeRepresentation("rgb")}

    with pytest.raises(AssertionError):
        _ = VRERuntimeArgs(video, representations, start_frame=0, end_frame=3, exception_mode="stop_execution",
                           n_threads_data_storer=0)
    with pytest.raises(AssertionError):
        _ = VRERuntimeArgs(video, representations, start_frame="lala", end_frame=3, exception_mode="stop_execution",
                           n_threads_data_storer=0)
    with pytest.raises(AssertionError):
        _ = VRERuntimeArgs(video, representations, start_frame="lala", end_frame=3, exception_mode="lala",
                           n_threads_data_storer=0)

    runtime_args = VRERuntimeArgs(video, representations, start_frame=0, end_frame=None, n_threads_data_storer=0,
                                  exception_mode="stop_execution")

    assert runtime_args is not None
    assert runtime_args.to_dict().keys() == {"video_path", "video_fps", "video_shape", "representations", "frames",
                                             "exception_mode", "n_threads_data_storer"}
