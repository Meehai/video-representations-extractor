from ngclib.nodes import RGB as _RGB
from nwmodule.graph import MapNode
from .Map2Map import EncoderMap2Map, DecoderMap2Map

class RGB(_RGB):
	def getEncoder(self, outputNode):
		assert isinstance(outputNode, MapNode)
		return EncoderMap2Map(dIn=self.numDims)

	def getDecoder(self, inputNode):
		assert False, "RGB cannot be decoded in this project."

	def aggregate(self):
		pass
	