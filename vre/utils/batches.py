"""utilities around batches in VRE"""
import numpy as np
from .vre_video import VREVideo
from ..logger import vre_logger as logger

def make_batches(video: VREVideo, start_frame: int, end_frame: int, batch_size: int) -> np.ndarray:
    """return 1D array [start_frame, start_frame+bs, start_frame+2*bs... end_frame]"""
    if batch_size > end_frame - start_frame:
        logger.warning(f"batch size {batch_size} is larger than #frames to process [{start_frame}:{end_frame}].")
        batch_size = end_frame - start_frame
    last_one = min(end_frame, len(video))
    batches = np.arange(start_frame, last_one, batch_size)
    batches = np.array([*batches, last_one], dtype=np.int64) if batches[-1] != last_one else batches
    return batches
