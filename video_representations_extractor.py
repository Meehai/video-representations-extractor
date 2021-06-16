import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from representations import Representation
from tqdm import trange
from media_processing_lib.image import tryWriteImage

class VideoRepresentationsExtractor:
	# @param[in] representations An topological sorted ordered dict (name, obj) with all instantiated representations
	# @param[in] exportCollage Whether to export a PNG file at each time step
	# @param[in] collageOrder A list with the order for the collage. If none provided, use the order of 1st param
	def __init__(self, representations:Dict[str, Representation], exportCollage:bool, \
        collageOutputDir:Optional[Path]=None, collageOrder:Optional[List[str]]=None, \
		rowsCols:Optional[Tuple[int, int]]=None):
		assert len(representations) > 0
		firstKey = list(representations.keys())[0]
		self.representations = representations
		self.exportCollage = exportCollage
		self.video = self.representations[firstKey].video
		self.outputResolution = self.representations[firstKey].outShape
		self.outputDir = Path(self.representations[firstKey].baseDir)
		self.rowsCols = rowsCols
		if exportCollage:
			assert not collageOutputDir is None
			if collageOutputDir is None:
				collageOutputDir = self.outputDir / "collage"
			if collageOrder is None:
				collageOrder = list(representations.keys())
			self.collageOutputDir = Path(collageOutputDir)
			self.collageOrder = collageOrder
	
	# Given a stack of N images, find the closest square X>=N*N and then remove rows 1 by 1 until it still fits X
	# Example: 9: 3*3; 12 -> 3*3 -> 3*4 (3 rows). 65 -> 8*8 -> 8*9. 73 -> 8*8 -> 8*9 -> 9*9
	@staticmethod
	def makeCollage(images:np.ndarray, rowsCols:Optional[Tuple[int, int]]=None) -> np.ndarray:
		images = np.array(images)
		assert images.dtype == np.uint8
		N = len(images)
		imageShape = images[0].shape

		def getSquareRowsColumns(N):
			x = int(np.sqrt(N))
			r, c = x, x
			# There are only 2 rows possible between x^2 and (x+1)^2 because (x+1)^2 = x^2 + 2*x + 1, thus we can add 2 columns
			#  at most. If a 3rd column is needed, then closest lower bound is (x+1)^2 and we must use that.
			if c * r < N:
				c += 1
			if c * r < N:
				r += 1
			assert (c + 1) * r > N and c * (r + 1) > N
			return r, c

		if isinstance(rowsCols, (tuple, list)):
			assert len(rowsCols) == 2
			r, c = rowsCols
		else:
			assert rowsCols is None
			r, c = getSquareRowsColumns(N)
		assert r * c >= N
		# Add black images if needed
		result = np.zeros((r * c, *imageShape), dtype=np.uint8)
		result[0 : N] = images
		result = result.reshape((r, c, *imageShape))
		result = np.concatenate(np.concatenate(result, axis=1), axis=1)
		return result

	def doExport(self, startIx:int=0, endIx:int=None):
		if endIx is None:
			endIx = len(self.video)
		print(("[VideoRepresentationsExtractor::doExport] Video: %s. Start frame: %d. End frame: %d. Output dir: " + \
			"%s. Output resolution: %s") % (self.video, startIx, endIx, self.outputDir, self.outputResolution))
		nameRepresentations = ", ".join(["'%s'" % x.name for x in self.representations.values()])
		print("[VideoRepresentationsExtractor::doExport] Representations (%d): %s" % \
			(len(self.representations), nameRepresentations))

		assert startIx < endIx and startIx >= 0
		for t in trange(startIx, endIx, desc="[VideoRepresentationsExtractor::doExport]"):
			finalOutputs = {}
			for name, representation in self.representations.items():
				finalOutputs[name] = representation[t]
		
			if self.exportCollage:
				images = [self.representations[k].makeImage(finalOutputs[k]) for k in self.collageOrder]
				images = VideoRepresentationsExtractor.makeCollage(images, self.rowsCols)
				outImagePath = self.collageOutputDir / ("%d.png" % t)
				tryWriteImage(images, str(outImagePath))
