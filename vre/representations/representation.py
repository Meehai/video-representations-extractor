from __future__ import annotations
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Union, Any
from functools import lru_cache

from media_processing_lib.image import image_resize
from media_processing_lib.video import MPLVideo

# Either a single np.ndarray or a dict of format {"data": np.ndarray, "extra": other stuff.}
RepresentationOutput = Union[np.ndarray, Dict[str, np.ndarray]]

# @brief Generic video/image representation
class Representation(ABC):
    def __init__(self, video: MPLVideo, name: str, dependencies: List[Representation]):
        assert isinstance(dependencies, (set, list))
        self.name = name
        self.dependencies = dependencies
        # This attribute is checked in the main vre object to see that this parent constructor was called properly.
        self.video = video
        self.setup()

    @abstractmethod
    def make(self, t: int) -> RepresentationOutput:
        """
        Main method of this representation. Calls the internal representation's logic to transform the current provided
        RGB frame of the attached video into the output representation.
        Note: The representation that is returned is guaranteed to be a float32 in the range of [0-1].
        Some representations may not be natively represented like that (i.e. optical flow being [-1:1]). The end-user
        must take care of this cases.

        The returned value is either a simple numpy array of the same shape as the video or a dict with two entries:
        {"data": np_data, "extra": {relevant info about this frame}}. The data is converted behind the scenes in the
        second representation if only a numpy array is provided with an empty 'extra' dict.

        This is also invoked for repr[t] and repr(t).
        """

    @abstractmethod
    def make_image(self, x: RepresentationOutput) -> np.ndarray:
        """Given the output of self.make(t), which a [0:1] float32 numpy array, return a [0:255] uint8 image"""

    def resize(self, x: RepresentationOutput, height: int, width: int, **resize_args) -> RepresentationOutput:
        """
        Resizes a representation output made from self.make(t). Info about time may be passed via 'extra' dict.
        Update this for more complex cases.
        """
        y = image_resize(x["data"], height=height, width=width, **resize_args)
        return {"data": y, "extra": x["extra"]}

    # @brief Method that should automate the entire download/instantiate/resolve any issues with a representation.
    #  Since this is called at every __call__, we should be careful to not instantiate objects for every frame.
    @abstractmethod
    def setup(self):
        pass

    def __getitem__(self, t: int) -> RepresentationOutput:
        return self.__call__(t)

    @lru_cache(maxsize=1000)
    def __call__(self, t: int) -> RepresentationOutput:
        # t = t % len(self.video) # TODO: needed?
        assert t >= 0 and t < len(self.video)
        # Get the raw result of this representation
        raw_result = self.make(t)
        raw_result = {"data": raw_result, "extra": {}} if isinstance(raw_result, np.ndarray) else raw_result
        data = raw_result["data"]
        assert data.dtype in (np.float32, np.uint8), f"{self.name}: Dtype: {data.dtype}"
        if data.dtype == np.float32:
            assert data.min() >= 0 and data.max() <= 1, f"{self.name}: Min: {data.min():.2f}. Max: {data.max():.2f}"
        return raw_result

    # TODO: see if this is needed for more special/faster resizes
    # def resizeRawData(self, rawData: np.ndarray) -> np.ndarray:
    #     interpolation = "nearest" if rawData.dtype == np.uint8 else "bilinear"
    #     # OpenCV bugs with uint8 and nearest, adding 255 values (in range [0-1])
    #     dtype = np.int32 if rawData.dtype == np.uint8 else rawData.dtype
    #     resizedData = image_resize(
    #         rawData.astype(dtype),
    #         height=self.outShape[0],
    #         width=self.outShape[1],
    #         only_uint8=False,
    #         interpolation=interpolation,
    #     )
    #     resizedData = resizedData.astype(rawData.dtype)
    #     return resizedData
