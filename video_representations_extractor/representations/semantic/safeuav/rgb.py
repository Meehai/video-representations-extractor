import torch as tr
from nwmodule import CriterionType
from nwmodule.metrics import MetricWrapper
from torchmetrics import Metric
from typing import List
from overrides import overrides

from .map_node import MapNode
from .Map2Map import EncoderMap2Map, DecoderMap2Map

class RGB(MapNode):
    def __init__(self, name: str = "rgb"):
        super().__init__(name = name, numDims = RGB.numDims())

    @overrides
    def getNodeMetrics(self) -> List[Metric]:
        return [
            MetricWrapper("RGB (L1 pixel)", RGB.RGBMetricL1Pixel),
            MetricWrapper("RGB (L2)", RGB.RGBMetricL2)
        ]

    @overrides
    def getNodeCriterion(self) -> CriterionType:
        return RGB.lossFn

    @staticmethod
    def numDims() -> int:
        return 3

    @staticmethod
    def lossFn(y, t):
        return ((y - t)**2).mean()

    @staticmethod
    def RGBMetricL1Pixel(y, t, **k):
        # Remap y and t from [0 : 1] to [0 : 255]
        yRgbOrig = y * 255
        tRgbOrig = t * 255
        return tr.abs(yRgbOrig - tRgbOrig).mean()

    @staticmethod
    def RGBMetricL2(y, t, **k):
        return ((y - t)**2).mean()

    def getEncoder(self, outputNode):
        assert isinstance(outputNode, MapNode)
        return EncoderMap2Map(dIn=self.numDims)

    def getDecoder(self, inputNode):
        assert isinstance(inputNode, MapNode)
        return DecoderMap2Map(dOut=self.numDims)
