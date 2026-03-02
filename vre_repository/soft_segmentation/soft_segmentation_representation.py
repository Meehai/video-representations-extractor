"""soft_segmentation_representation.py - Generic class for all the soft segmentation algorithms"""
from overrides import overrides
from vre.representations import Representation
from vre.representations.mixins import NpIORepresentation, ResizableRepresentationMixin, NormedRepresentationMixin

class SoftSegmentationRepresentation(Representation, NpIORepresentation,
                                     NormedRepresentationMixin, ResizableRepresentationMixin):
    def __init__(self, name: str, **kwargs):
        Representation.__init__(self, name, **kwargs)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)
        ResizableRepresentationMixin.__init__(self)

    @overrides
    def n_channels(self):
        return 3
