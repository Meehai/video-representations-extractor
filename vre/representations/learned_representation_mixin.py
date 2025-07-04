"""
Helper mixin class that adds the weights/device relevant methods & properties for such representation
A representation that inherits this also must have weights in the weights repository.
"""
from abc import abstractmethod, ABC
import torch as tr

from vre.logger import vre_logger as logger

VREDevice = str | tr.device # not only torch, but this is what we support atm

class LearnedRepresentationMixin(ABC):
    """Learned Representastion Mixin for VRE implementation"""
    def __init__(self):
        self._device: VREDevice | None = None
        self.setup_called = False

    @abstractmethod
    def vre_setup(self, load_weights: bool = True):
        """
        Setup method for this representation. This is required to run this representation from within VRE. We do this
        setup separately, so we can instatiate the object without doing any VRE specific setup, like loading weights.
        """

    @abstractmethod
    def vre_free(self):
        """Needed to deallocate stuff from cuda mostly. After this, you need to run vre_setup() again."""

    @staticmethod
    @abstractmethod
    def weights_repository_links(**kwargs) -> list[str]:
        """Returns a name and a list of links to the weightss. The names are first checked in the local resources dir"""

    @property
    def device(self) -> VREDevice:
        """Returns the device of the representation"""
        if self._device is None:
            logger.warning(f"[{self}] No device set, returning 'cpu'. Call set_learned_params")
            return "cpu"
        return self._device

    @device.setter
    def device(self, dev: VREDevice):
        assert isinstance(dev, (str, tr.device)), dev
        self._device = dev

    def set_learned_params(self, **kwargs):
        """set the learned parameters for the representation"""
        attributes = ["device"]
        res = ""
        for attr in attributes:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
                res += f"\n-{attr}: {kwargs[attr]}"

        if len(res) > 0:
            logger.debug(f"[{self}] Set node specific 'Learned' params:{res}")
