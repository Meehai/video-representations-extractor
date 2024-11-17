"""Init file"""
# pylint: disable=reimported, wrong-import-position
import warnings
from diffusers.utils.logging import disable_progress_bar
import matplotlib
try:
    matplotlib.use("TkAgg") # sometimes we default to PyQt5 if this is not set here which crashes randomly....
except ImportError:
    matplotlib.use("Agg")
from .video_representations_extractor import VideoRepresentationsExtractor, VideoRepresentationsExtractor as VRE
from .representations import Representation, ReprOut
from .utils.vre_video import FFmpegVideo
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
