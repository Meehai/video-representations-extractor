import numpy as np
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from tqdm import trange
from media_processing_lib.image import collageFn, tryWriteImage
from media_processing_lib.video import MPLVideo
from nwutils.others import topologicalSort
from collections import OrderedDict
from functools import partial

from .representations import Representation, getRepresentation

# Function that validates the output dir (args.outputDir) and each representation. By default, it just dumps the yaml
#  file of the representation under outputDir/representationName (i.e. video1/rgb/cfg.yaml). However, if the directory
#  outputDir/representationName exists, we check that the cfg file is identical to the gloal cfg file (args.cfgPath)
#  and we check that _all_ npz files have been computted. If so, we can safely skip this representation to avoid doing
#  duplicate work. If either the number of npz files is wrong or the cfg file is different, we throw and error and the
#  user must change the global cfg file for that particular representation: either by changing the resulting name or
#  updating the representation's parameters accordingly.
def makeOutputDirs(cfg:Dict, outputDir:Path, outputResolution:Tuple[int,int]):
	outputDir.mkdir(parents=True, exist_ok=True)
	for name, values in cfg.items():
		Dir = outputDir / name
		thisCfgPath = Dir / "cfg.yaml"
		if isinstance(values, Representation):
			values = {"method":values.name, "dependencies":values.dependencies, "parameters":values.getParameters()}
		for v in values["dependencies"]:
			assert isinstance(v, str), v

		thisCfg = {name:values, "outputResolution":outputResolution}
		if Dir.exists() and thisCfgPath.exists():
			loadedCfg = yaml.safe_load(open(thisCfgPath, "r"))
			if loadedCfg != thisCfg:
				yaml.safe_dump(thisCfg, open(thisCfgPath, "w"))
				loadedCfg = yaml.safe_load(open(thisCfgPath, "r"))
			assert loadedCfg == thisCfg, f"Wrong cfg file.\n - Loaded: {loadedCfg}.\n - This: {thisCfg}"
		else:
			Dir.mkdir(exist_ok=True)
			yaml.safe_dump(thisCfg, open(thisCfgPath, "w"))

def topoSortRepresentations(representations:Dict[str, Union[str, Representation]]) -> List[str]:
	print("[VideoRepresentationsExtractor::topoSortRepresentations] Doing topological sort...")
	depGraph = {}
	for k in representations:
		if isinstance(representations[k], Representation):
			depGraph[k] = representations[k].dependencies
		else:
			depGraph[k] = representations[k]["dependencies"]
	topoSort = topologicalSort(depGraph)
	topoSort = OrderedDict({k : representations[k] for k in topoSort})
	return topoSort

class VideoRepresentationsExtractor:
	# @param[in] representations An not topological sorted ordered dict (name, obj/str). If they are str, we assume
	#  that they are uninstantiated built-in layers, so we use getRepresentation. Otherwise, we assume they are custom
	#  preinstantiated representations (see 3Depths project), so we just use them as is.
	# @param[in] exportCollage Whether to export a PNG file at each time step
	# @param[in] collageOrder A list with the order for the collage. If none provided, use the order of 1st param
	def __init__(self, video:MPLVideo, outputDir:Path, outputResolution:Tuple[int, int], \
		representations:Dict[str, Union[str, Representation]]):
		assert len(representations) > 0
		outputDir = Path(outputDir)
		makeOutputDirs(representations, outputDir, outputResolution)
		topoSortedRepresentations = topoSortRepresentations(representations)

		self.video = video
		self.outputResolution = outputResolution
		self.outputDir = outputDir

		# Topo sorted and not topo sorted representations (for collage)
		self.tsr = self.doInstantiation(topoSortedRepresentations)
		self.representations = {k : self.tsr[k] for k in representations}

	def doInstantiation(self, topoSortedRepresentations) -> Dict[str, Representation]:
		res = {}
		for name in topoSortedRepresentations:
			r = topoSortedRepresentations[name]
			if isinstance(r, Representation):
				print(f"Representation='{r.name}' already instantiated. Skipping.")
				obj = r
				# If they are already instantiated, we may send both strings to methods or the objects themselves.
				obj.dependencies = [res[k] if isinstance(k, str) else k for k in obj.dependencies]
			else:
				assert isinstance(r, dict)
				print(f"Representation='{name}'. Instantiating...")

				dependencies = [res[k] for k in r["dependencies"]]
				# If we have aliases, use these names, otherwise, use the representation's name itself.
				dependencyAliases = r["dependencyAliases"] if "dependencyAliases" in r else r["dependencies"]
				assert "type" in r and "method" in r, f"Broken format: {r.keys()}"
				objType = getRepresentation(r["type"], r["method"])
				objType = partial(objType, name=name, dependencies=dependencies, dependencyAliases=dependencyAliases)
				# The representation parameters.
				objParams = r["parameters"] if not r["parameters"] is None else {}
				# saveResults in yaml cfg can be "none", "all", "resized_only" and defaults to "all" if not present.
				objParams["saveResults"] = r["saveResults"] if "saveResults" in r else "all"
				obj = objType(**objParams)

			obj.setVideo(self.video)
			obj.setBaseDir(self.outputDir)
			obj.setOutShape(self.outputResolution)
			res[name] = obj
		return res

	def doExport(self, startIx:int=0, endIx:int=None, exportCollage:bool=True, \
			collageOrder:List[str]=None, collageDirStr:str="collage"):
		if endIx is None:
			endIx = len(self.video)

		print(f"\n - Video: {self.video} \n - Start frame: {startIx}. " + \
			f"End frame: {endIx}\n - Output dir: {self.outputDir} \n - Output resolution: {self.outputResolution}")
		nameRepresentations = ", ".join([f"'{x.name}'" for x in self.representations.values()])
		print(f"Representations ({len(self.representations)}): {nameRepresentations}")

		if exportCollage:
			collageOrder = list(self.representations.keys()) if collageOrder is None else collageOrder
			collageOutputDir = self.outputDir / collageDirStr
			collageOutputDir.mkdir(exist_ok=True)
			print(f"Exporting collage to: {collageOutputDir}")

		assert startIx < endIx and startIx >= 0
		for t in trange(startIx, endIx, desc="[VideoRepresentationsExtractor::doExport]"):
			finalOutputs = {}
			for name, representation in self.tsr.items():
				finalOutputs[name] = representation[t]
		
			if exportCollage:
				images = [self.representations[k].makeImage(finalOutputs[k]) for k in collageOrder]
				images = collageFn(images, titles=collageOrder)
				outImagePath = collageOutputDir / f"{t}.png"
				tryWriteImage(images, str(outImagePath))
