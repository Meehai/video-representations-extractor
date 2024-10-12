"""
Helper mixin class that adds the weights/device relevant methods & properties for such representation
A representation that inherits this also must have weights in the weights repository.
"""
from abc import abstractmethod, ABC
import torch as tr

VREDevice = str | tr.device # not only torch, but this is what we support atm

class LearnedRepresentationMixin(ABC):
    """Learned Representastion Mixin for VRE implementation"""
    def __init__(self):
        self.device: VREDevice = "cpu"

    # TODO: make this and get rid of the hardcoded stuff in utils.
    # @property
    # @abstractmethod
    # def weights_files(self) -> list[str]:
    #     """
    #     A list of files that must be present on the disk or downloaded from the weights repository. Only the stem is
    #     needed and vre_setup() must handle their loading. The data is stored under {weights_dir}/{repr_type}/[names]
    #     For example: 'depth/dpt' has 'depth_dpt_midas.pth' and is stored at 'weights_dir/depth/dpt/depth_dpt_midas.pth
    #     """

    @abstractmethod
    def vre_setup(self, load_weights: bool = True):
        """
        Setup method for this representation. This is required to run this representation from within VRE. We do this
        setup separately, so we can instatiate the object without doing any VRE specific setup, like loading weights.
        """

    @abstractmethod
    def vre_free(self):
        """Needed to deallocate stuff from cuda mostly. After this, you need to run vre_setup() again."""
