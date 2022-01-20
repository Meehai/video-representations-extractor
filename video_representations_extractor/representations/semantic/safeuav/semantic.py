from ngclib.nodes import Semantic as _Semantic
from nwgraph import MapNode
from .Map2Map import EncoderMap2Map, DecoderMap2Map

class Semantic(_Semantic):
	def getEncoder(self, outputNode):
		assert isinstance(outputNode, MapNode)
		return EncoderMap2Map(dIn=self.numDims)

	def getDecoder(self, inputNode):
		assert isinstance(inputNode, MapNode)
		return DecoderMap2Map(dOut=self.numDims)
