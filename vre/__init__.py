"""Init file"""
# pylint: disable=reimported, wrong-import-position
import warnings
from .video_representations_extractor import VideoRepresentationsExtractor, VideoRepresentationsExtractor as VRE
from .representations import Representation
from .utils.lovely import monkey_patch
from .utils.repr_memory_layout import MemoryData, DiskData
from .utils.repr_out import ReprOut

try:
    import pdbp
except ImportError:
    pass

monkey_patch()

try:
    import torch # noqa: F401
    warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
except ImportError:
    pass
