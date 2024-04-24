"""VRE representation mixin. This is used to integrate representations with VRE."""
import numpy as np
from .utils import VREVideo

RepresentationOutput = np.ndarray | tuple[np.ndarray, list[dict]]

class VRERepresentationMixin:
    """VRE representation mixin. This is used to integrate representations with VRE."""
    def vre_setup(self, video: VREVideo, **kwargs):
        """
        Setup method for this representation. This is required to run this representation from within VRE.
        We do this setup separately, so we can instatiate the representation without doing any VRE specific setup,
        like loading weights.
        """

    # pylint: disable=unused-argument
    def vre_dep_data(self, video: VREVideo, ix: slice) -> dict[str, RepresentationOutput]:
        """method used to retrieve the dependencies' data for this frames during a vre run"""
        return {}

    def vre_make(self, video: VREVideo, ix: slice, make_images: bool) -> (RepresentationOutput, np.ndarray | None):
        """
        Method used to integrate with VRE. Gets the entire data (video) and a slice of it (ix) and returns the
        representation for that slice. Additionally, if makes_images is set to True, it also returns the image
        representations of this slice.
        """
        print("A")
        frames = np.array(video[ix])
        print("B")
        dep_data = self.vre_dep_data(video, ix)
        print("C")
        res = self.make(frames, **dep_data)
        print("D")
        repr_data, extra = res if isinstance(res, tuple) else (res, {})
        print("E")
        imgs = self.make_images(frames, res) if make_images else None
        print("F")
        return (repr_data, extra), imgs
