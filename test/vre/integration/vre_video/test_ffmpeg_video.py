import numpy as np
from vre import FFmpegVideo
from vre.utils import fetch_resource

def test_FFmpegVideo_getitem():
    video = FFmpegVideo(fetch_resource("test_video.mp4"))
    frame = video[np.random.randint(0, len(video))]
    assert isinstance(frame, np.ndarray) and frame.shape == (720, 1280, 3)

    frames = video[np.random.randint(0, len(video), size=5)]
    assert frames.shape == (5, 720, 1280, 3)
