from __future__ import annotations
import numpy as np
from pathlib import Path
from overrides import overrides
from typing import Dict, Tuple, Callable
from media_processing_lib.video import MPLVideo

from .representation import Representation

# @brief A representation that is a placeholder for a set of npz files that were precomputted by other (not yet
#  not yet supported) algorithms, such as Sfm, SemanticGB, weird pre-trained networks etc. The only thing it needs is
#  to provide the correct files (0.npz, ..., N.npz) as well as a function to plot the data to human viewable format.
class DummyRepresentation(Representation):
	def __init__(self, baseDir:Path, name:str, dependencies:Dict[str, Representation], \
		video:MPLVideo, outShape:Tuple[int, int], makeImageFn:Callable):
		super().__init__(baseDir, name, dependencies, video, outShape)
		assert isinstance(makeImageFn, Callable)
		self.makeImageFn = makeImageFn

	@overrides
	def make(self, t:int) -> np.ndarray:
		assert False, "Dummy representation (%s) has no precomputted npz files!" % self.name
	
	@overrides
	def makeImage(self, x:Dict) -> np.ndarray:
		return self.makeImageFn(x)

	@overrides
	def setup(self):
		pass
