"""Init file"""
# pylint: disable=reimported, wrong-import-position
import warnings
import sys
from pathlib import Path
from diffusers.utils.logging import disable_progress_bar


from .video_representations_extractor import VideoRepresentationsExtractor, VideoRepresentationsExtractor as VRE
from .representations import Representation, ReprOut
from .vre_video import VREVideo, FFmpegVideo, FakeVideo
from .utils.lovely import monkey_patch
from .utils.repr_memory_layout import MemoryData, DiskData

try:
    import pdbp
except ImportError:
    pass

monkey_patch()
disable_progress_bar()

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
warnings.filterwarnings("ignore", "`clean_up_tokenization_spaces` was not set*.")
