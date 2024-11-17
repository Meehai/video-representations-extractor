"""Init file"""
# pylint: disable=reimported
from .build_representations import (
    build_representations_from_cfg, build_representation_from_cfg, add_external_representations)
from .representation import Representation, ReprOut
from .task_mapper import TaskMapper

from .learned_representation_mixin import LearnedRepresentationMixin
from .compute_representation_mixin import ComputeRepresentationMixin
from .normed_representation_mixin import NormedRepresentationMixin

from .io_representation_mixin import IORepresentationMixin
from .np_io_representation import NpIORepresentation
