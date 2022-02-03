from __future__ import annotations
from typing import Optional, Dict, Union, Any, Set, List
from abc import ABC, abstractmethod
import torch as tr
import torch.nn as nn
from torchmetrics import Metric

class Node(ABC):
    def __init__(self, name: str, hyperParameters: dict={}):
        super().__init__()
        self.name = name

        # Set up hyperparameters for this node (used for saving/loading identical node)
        self.hyperParameters = self.getHyperParameters(hyperParameters)
        # Messages are the items received at this node via all its incoming edges.
        self.messages: Set = set()

    @abstractmethod
    def getEncoder(self, outputNode: Optional[Node] = None) -> nn.Module:
        pass

    @abstractmethod
    def getDecoder(self, inputNode: Optional[Node] = None) -> nn.Module:
        pass

    @abstractmethod
    def getNodeMetrics(self):
        pass

    @abstractmethod
    def getNodeCriterion(self):
        pass

    def clearMessages(self):
        self.messages = set()

    def addMessage(self, message):
        pass

    def getMessages(self):
        pass

    def getHyperParameters(self, hyperParameters: Dict) -> Dict:
        # This is some weird bug. If i leave the same hyperparameters coming (here I make a shallow copy),
        #  making two instances of the same class results in having same hyperparameters.
        hyperParameters = {k: hyperParameters[k] for k in hyperParameters.keys()}
        hyperParameters["name"] = self.name
        return hyperParameters

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    # This and __eq__ are used so we can put node in dict and access them via strings
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, x) -> bool:
        if isinstance(x, Node):
            x = x.name
        return self.name == x


class MapNode(Node):
    """Generic Map Node (2D) having a number of channels (D1xD2xNC)"""
    def __init__(self, name: str, numDims: int, hyperParameters: dict={}):
        Node.__init__(self, name, hyperParameters)
        self.numDims = numDims
