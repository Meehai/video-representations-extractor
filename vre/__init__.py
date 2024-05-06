"""Init file"""
from lovely_tensors import monkey_patch

# pylint: disable=reimported
from .video_representations_extractor import VideoRepresentationsExtractor, VideoRepresentationsExtractor as VRE

monkey_patch()
