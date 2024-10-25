from vre.utils import FakeVideo
from vre.vre_runtime_args import VRERuntimeArgs
from vre.representations.fake_representation import FakeRepresentation
import numpy as np
import pytest

def test_VRERuntimeArgs_ctor():
    video = FakeVideo(np.random.randint(0, 255, size=(2, 128, 128, 3), dtype=np.uint8), frame_rate=30)
    representations = {"rgb": FakeRepresentation("rgb")}

    with pytest.raises(AssertionError):
        _ = VRERuntimeArgs(video, representations, start_frame=0, end_frame=3, batch_size=5,
                           exception_mode="stop_execution", output_size="native", load_from_disk_if_computed=False)
        _ = VRERuntimeArgs(video, representations, start_frame="lala", end_frame=3, batch_size=5,
                           exception_mode="stop_execution", output_size="native", load_from_disk_if_computed=False)
        _ = VRERuntimeArgs(video, representations, start_frame="lala", end_frame=3, batch_size=5,
                           exception_mode="lala", output_size="native", load_from_disk_if_computed=False)
        _ = VRERuntimeArgs(video, representations, start_frame="lala", end_frame=3, batch_size=5,
                           exception_mode="stop_execution", output_size="lala", load_from_disk_if_computed=False)

    runtime_args = VRERuntimeArgs(video, representations, start_frame=0, end_frame=None, batch_size=5,
                                  exception_mode="stop_execution", output_size="native",
                                  load_from_disk_if_computed=False)

    assert runtime_args is not None
    assert runtime_args.batch_sizes["rgb"] == 2 # 2 < 5

    runtime_args = VRERuntimeArgs(video, representations, start_frame=0, end_frame=None, batch_size=5,
                                  exception_mode="stop_execution", output_size="video_shape",
                                  load_from_disk_if_computed=False)
    assert runtime_args.output_sizes["rgb"] == (128, 128)
