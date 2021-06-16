import numpy as np
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from tqdm import trange
from media_processing_lib.image import tryWriteImage
from media_processing_lib.video import MPLVideo
from nwdata.utils import topologicalSort
from collections import OrderedDict
from functools import partial

from representations import Representation, getRepresentation

# Function that validates the output dir (args.outputDir) and each representation. By default, it just dumps the yaml
#  file of the representation under outputDir/representationName (i.e. video1/rgb/cfg.yaml). However, if the directory
#  outputDir/representationName exists, we check that the cfg file is identical to the gloal cfg file (args.cfgPath)
#  and we check that _all_ npz files have been computted. If so, we can safely skip this representation to avoid doing
#  duplicate work. If either the number of npz files is wrong or the cfg file is different, we throw and error and the
#  user must change the global cfg file for that particular representation: either by changing the resulting name or
#  updating the representation's parameters accordingly.
def makeOutputDirs(cfg:Dict, outputDir:Path, outputResolution:Tuple[int,int], exportCollage:bool):
	outputDir.mkdir(parents=True, exist_ok=True)
	if exportCollage:
		(outputDir / "collage").mkdir(exist_ok=True)
	for name, values in cfg.items():
		Dir = outputDir / name
		thisCfgPath = Dir / "cfg.yaml"
		thisCfg = {name:values, "outputResolution":outputResolution}
		if Dir.exists() and thisCfgPath.exists():
			loadedCfg = yaml.safe_load(open(thisCfgPath, "r"))
			assert loadedCfg == thisCfg, "Wrong cfg file.\n - Loaded: %s.\n - This: %s" % (loadedCfg, thisCfg)
		else:
			Dir.mkdir(exist_ok=True)
			yaml.safe_dump(thisCfg, open(thisCfgPath, "w"))

def topoSortRepresentations(representations:Dict[str, Union[str, Representation]]) -> List[str]:
	print("[VideoRepresentationsExtractor::topoSortRepresentations] Doing topological sort...")
	depGraph = {}
	for k in representations:
		if isinstance(representations[k], Representation):
			depGraph[k] = list(representations[k].dependencies.keys())
		else:
			depGraph[k] = representations[k]["dependencies"]
	topoSort = topologicalSort(depGraph)
	topoSort = OrderedDict({k : representations[k] for k in topoSort})
	return topoSort

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

class VideoRepresentationsExtractor:
	# @param[in] representations An not topological sorted ordered dict (name, obj/str). If they are str, we assume
	#  that they are uninstantiated built-in layers, so we use getRepresentation. Otherwise, we assume they are custom
	#  preinstantiated representations (see 3Depths project), so we just use them as is.
	# @param[in] exportCollage Whether to export a PNG file at each time step
	# @param[in] collageOrder A list with the order for the collage. If none provided, use the order of 1st param
	def __init__(self, video:MPLVideo, outputDir:Path, outputResolution:Tuple[int, int], \
		representations:Dict[str, Union[str, Representation]], exportCollage:bool, \
		collageOrder:Optional[List[str]]=None, rowsCols:Optional[Tuple[int, int]]=None):
		assert len(representations) > 0
		makeOutputDirs(representations, outputDir, outputResolution, exportCollage)
		topoSortedRepresentations = topoSortRepresentations(representations)

		self.exportCollage = exportCollage
		self.video = video
		self.outputResolution = outputResolution
		self.outputDir = outputDir

		self.rowsCols = rowsCols if not rowsCols is None else getSquareRowsColumns(len(representations))
		self.collageOutputDir = self.outputDir / "collage"
		self.collageOrder = collageOrder = list(representations.keys())

		# Topo sorted and not topo sorted representations (for collage)
		self.tsr = self.doInstantiation(topoSortedRepresentations)
		self.representations = {k : self.tsr[k] for k in representations}

	def doInstantiation(self, topoSortedRepresentations) -> Dict[str, Representation]:
		res = {}
		for name in topoSortedRepresentations:
			r = topoSortedRepresentations[name]
			if isinstance(r, Representation):
				print(("[VideoRepresentationsExtractor::doInstantation] Representation='%s' already ") + \
					("instantiated. Skipping.") % name)
				obj = r
			else:
				assert isinstance(r, dict)
				print("[VideoRepresentationsExtractor::doInstantation] Representation='%s'. Instantiating..." % name)

				dependencies = {k : res[k] for k in r["dependencies"]}
				objType = getRepresentation(r["method"])
				objType = partial(objType, name=name, dependencies=dependencies)
				obj = objType(**r["parameters"]) if not r["parameters"] is None else objType()
			obj.setVideo(self.video)
			obj.setBaseDir(self.outputDir)
			obj.setOutShape(self.outputResolution)
			res[name] = obj
		return res

	# Given a stack of N images, find the closest square X>=N*N and then remove rows 1 by 1 until it still fits X
	# Example: 9: 3*3; 12 -> 3*3 -> 3*4 (3 rows). 65 -> 8*8 -> 8*9. 73 -> 8*8 -> 8*9 -> 9*9
	@staticmethod
	def makeCollage(images:np.ndarray, rowsCols:Optional[Tuple[int, int]]=None) -> np.ndarray:
		images = np.array(images)
		assert images.dtype == np.uint8
		N = len(images)
		imageShape = images[0].shape

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
		print(("[VideoRepresentationsExtractor::doExport]\n - Video: %s \n - Start frame: %d. End frame: %d " + \
			"\n - Output dir: %s \n - Output resolution: %s") % \
			(self.video, startIx, endIx, self.outputDir, self.outputResolution))
		nameRepresentations = ", ".join(["'%s'" % x.name for x in self.representations.values()])
		print("[VideoRepresentationsExtractor::doExport] Representations (%d): %s" % \
			(len(self.representations), nameRepresentations))

		assert startIx < endIx and startIx >= 0
		for t in trange(startIx, endIx, desc="[VideoRepresentationsExtractor::doExport]"):
			finalOutputs = {}
			for name, representation in self.tsr.items():
				finalOutputs[name] = representation[t]
		
			if self.exportCollage:
				images = [self.representations[k].makeImage(finalOutputs[k]) for k in self.collageOrder]
				images = VideoRepresentationsExtractor.makeCollage(images, self.rowsCols)
				outImagePath = self.collageOutputDir / ("%d.png" % t)
				tryWriteImage(images, str(outImagePath))
