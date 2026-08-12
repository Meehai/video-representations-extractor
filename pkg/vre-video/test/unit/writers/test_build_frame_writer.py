import pytest
from vre_video.writers import build_frame_writer, PILFrameWriter, NumpyFrameWriter

def test_build_frame_writer_PILFrameWriter():
    assert isinstance(build_frame_writer("PIL"), PILFrameWriter)
    assert isinstance(build_frame_writer("pillow"), PILFrameWriter)
    with pytest.raises(NotImplementedError):
        _ = build_frame_writer("pil")
    assert isinstance(build_frame_writer(build_frame_writer("pillow")), PILFrameWriter)
    assert isinstance(build_frame_writer(build_frame_writer(None)), PILFrameWriter) # default

    writer: PILFrameWriter = build_frame_writer("PIL")
    assert writer.suffix == "png"
    writer: PILFrameWriter = build_frame_writer("PIL", fmt="jpg")
    assert writer.suffix == "jpg"

def test_build_frame_writer_NumpyFrameWriter():
    assert isinstance(build_frame_writer("numpy"), NumpyFrameWriter)
    assert isinstance(build_frame_writer("np"), NumpyFrameWriter)
    assert isinstance(build_frame_writer(build_frame_writer("numpy")), NumpyFrameWriter)

    writer: NumpyFrameWriter = build_frame_writer("numpy")
    assert writer.format == "npy"
    assert writer.compress is False
    writer: NumpyFrameWriter = build_frame_writer("numpy", fmt="npz")
    assert writer.format == "npz"
    assert writer.compress is False
    writer: NumpyFrameWriter = build_frame_writer("numpy", fmt="npz", compress=True)
    assert writer.format == "npz"
    assert writer.compress is True
