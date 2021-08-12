import torch as tr
from ngclib.nodes import Semantic as _Semantic
from ngclib.voting_algorithms import simpleMean
from nwmodule.graph import MapNode, Message
from .Map2Map import EncoderMap2Map, DecoderMap2Map

class Semantic(_Semantic):
	def __init__(self, semanticClasses, semanticColors, name:str, useGlobalMetrics:bool=False):
		super().__init__(semanticClasses, name, useGlobalMetrics)
		self.semanticColors = semanticColors
		assert len(self.semanticClasses) == len(self.semanticColors)

	def getEncoder(self, outputNode):
		assert isinstance(outputNode, MapNode)
		return EncoderMap2Map(dIn=self.numDims)

	def getDecoder(self, inputNode):
		assert isinstance(inputNode, MapNode)
		return DecoderMap2Map(dOut=self.numDims)

	def aggregate(self):
		messages = self.getMessages()
		newMessage = simpleMean(messages)
		self.clearMessages()
		self.addMessage(newMessage)
