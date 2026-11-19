"""descriptor_representation.py - Defines the concepts of keypoints (if it uses) and descriptors (or embeddings)"""
from abc import ABC
from vre import Representation
from vre.representations.mixins import NpIORepresentationMixin

class DescriptorRepresentation(NpIORepresentationMixin, Representation, ABC):
    """Defines the concepts of keypoints (if it uses) and descriptors (or embeddings)"""
