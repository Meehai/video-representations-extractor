from typing import List, Union, Tuple
from overrides import overrides
from nwmodule.loss import softmax_nll

from .map_node import MapNode
from .Map2Map import EncoderMap2Map, DecoderMap2Map

class Semantic(MapNode):
    def __init__(self, semanticClasses: Union[int, List[str]], semanticColors: List[Tuple[int, int, int]], \
            name: str="semantic", useGlobalMetrics: bool=False):

        if isinstance(semanticClasses, int):
            semanticClasses = list(range(semanticClasses))
        assert isinstance(semanticClasses, (list, tuple))

        super().__init__(name=name, numDims=len(semanticClasses))
        self.semanticClasses = semanticClasses
        self.semanticColors = semanticColors
        self.numClasses = len(semanticClasses)
        self.useGlobalMetrics = useGlobalMetrics
        assert len(self.semanticClasses) == len(self.semanticColors), f"{self.semanticClasses} " + \
            f"({len(self.semanticClasses)}) vs {self.semanticColors} ({len(self.semanticColors)})"

    @overrides
    def getNodeMetrics(self):
        return {}

    @overrides
    def getNodeCriterion(self):
        return Semantic.lossFn

    @staticmethod
    def lossFn(y, t):
        return softmax_nll(y, t, dim=-1).mean()

    def getEncoder(self, outputNode):
        assert isinstance(outputNode, MapNode)
        return EncoderMap2Map(dIn=self.numDims)

    def getDecoder(self, inputNode):
        assert isinstance(inputNode, MapNode)
        return DecoderMap2Map(dOut=self.numDims)
