"""Helper mixin class that adds the VRE relevant methods & properties such that a representation works in vre loop"""
from .utils import parsed_str_type, VREVideo, RepresentationOutput
from .logger import logger

class VRERepresentationMixin:
    """VRERepresentationMixin class"""
    def __init__(self):
        self.vre_parameters = {}
        self.batch_size: int | None = None
        self.output_size: tuple[int, int] | str | None = None

    # pylint: disable=unused-argument
    def vre_setup(self, video: VREVideo, **kwargs):
        """
        Setup method for this representation. This is required to run this representation from within VRE.
        We do this setup separately, so we can instatiate the representation without doing any VRE specific setup,
        like loading weights.
        Parameters:
        - video The video we are working with during the VRE run
        - kwargs Any other parameters that can be passed via `build_representation`
        """
        logger.debug(f"[{parsed_str_type(self)}] No runtime setup provided.")

    # pylint: disable=unused-argument
    def vre_dep_data(self, video: VREVideo, ix: slice) -> dict[str, RepresentationOutput]:
        """method used to retrieve the dependencies' data for this frames during a vre run"""
        return {}
