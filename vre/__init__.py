"""Init file"""
import warnings
from lovely_tensors import monkey_patch
from diffusers.utils.logging import disable_progress_bar

# pylint: disable=reimported
from .video_representations_extractor import VideoRepresentationsExtractor, VideoRepresentationsExtractor as VRE

monkey_patch()
disable_progress_bar()

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
warnings.filterwarnings("ignore", "`clean_up_tokenization_spaces` was not set*.")
