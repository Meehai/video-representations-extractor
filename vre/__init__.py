"""Init file"""
import warnings
from lovely_tensors import monkey_patch

# pylint: disable=reimported
from .video_representations_extractor import VideoRepresentationsExtractor, VideoRepresentationsExtractor as VRE

monkey_patch()

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
