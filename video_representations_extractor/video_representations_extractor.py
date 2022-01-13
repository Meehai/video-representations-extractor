from os import statvfs_result
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from tqdm import trange
from media_processing_lib.image import collage_fn, image_write
from media_processing_lib.video import MPLVideo, video_read
from nwutils.others import topologicalSort
from collections import OrderedDict
from functools import partial

from .representations import Representation, getRepresentation
from .logger import logger

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
			values = {
				"type":None,
				"method":values.name,
				"dependencies":values.dependencies,
				"parameters":values.getParameters()
			}
		assert values is not None and "dependencies" in values and "method" in values \
			and "type" in values and "parameters" in values, f"Malformed input for {name}: {values}"
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
	logger.debug("Doing topological sort...")
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
	def __init__(self, video:Union[str, Path, MPLVideo], outputDir:Path, \
			representations:Dict[str, Union[str, Representation]], outputResolution:Optional[Tuple[int, int]]=None):
		if isinstance(video, (str, Path)):
			logger.info(f"Path '{video}' provided, reading video using pims.")
			video = video_read(video, vid_lib="pims")

		assert len(representations) > 0
		outputDir = Path(outputDir)
		if outputResolution is None:
			outputResolution = list(video.shape[1:3])
			logger.info(f"Output resolution not set. Infering from video shape: {outputResolution}")
		makeOutputDirs(representations, outputDir, outputResolution)

		self.video = video
		self.outputResolution = outputResolution
		self.outputDir = outputDir

		# Topo sorted and not topo sorted representations (for collage)
		topoSortedRepresentations = topoSortRepresentations(representations)
		self.tsr = self.doInstantiation(topoSortedRepresentations)
		self.representations = {k : self.tsr[k] for k in representations}

	def doInstantiation(self, topoSortedRepresentations) -> Dict[str, Representation]:
		res = {}
		for name in topoSortedRepresentations:
			r = topoSortedRepresentations[name]
			if isinstance(r, Representation):
				logger.debug(f"Representation='{r.name}' already instantiated. Skipping.")
				obj = r
				# If they are already instantiated, we may send both strings to methods or the objects themselves.
				obj.dependencies = [res[k] if isinstance(k, str) else k for k in obj.dependencies]
			else:
				logger.debug(f"Representation='{name}'. Instantiating...")
				assert isinstance(r, dict), f"Broken format (not a dict) for {name}. Type: {type(r)}."
				assert "type" in r and "method" in r, f"Broken format: {r.keys()}"

				dependencies = [res[k] for k in r["dependencies"]]
				# If we have aliases, use these names, otherwise, use the representation's name itself.
				if "dependencyAliases" in r:
					dependencyAliases = r["dependencyAliases"]
				else:
					logger.debug2("Dependency aliases not in cfg, defaulting to the depdenceny names.")
					dependencyAliases = r["dependencies"]
				assert len(dependencyAliases) == len(dependencies)

				# If we have save results, use that, instead, use 'all'.
				if "saveResults" in r:
					saveResults = r["saveResults"]
				else:
					logger.debug2("Save results not in cfg, defaulting to 'all'.")
					saveResults = "all"
				assert saveResults in ("all", "none", "resized_only"), f"Got: '{saveResults}'."

				objType = getRepresentation(r["type"], r["method"])
				vreParams = {"name": name, "dependencies": dependencies, \
					"dependencyAliases": dependencyAliases, "saveResults": saveResults}
				parameters = r["parameters"] if not r["parameters"] is None else {}
				obj = objType(**parameters, **vreParams)

			obj.setVideo(self.video)
			obj.setBaseDir(self.outputDir)
			obj.setOutShape(self.outputResolution)
			assert obj.instantiated == True, f"Object {obj} (name: '{name}') not properly instantiated!"
			res[name] = obj
		return res

	def doExport(self, startIx:int=0, endIx:int=None, exportCollage:bool=True, \
			collageOrder:List[str]=None, collageDirStr:str="collage"):
		if endIx is None:
			endIx = len(self.video)

		logger.info(f"\n{self}\n  - Start frame: {startIx}. End frame: {endIx}.\n")

		if exportCollage:
			collageOrder = list(self.representations.keys()) if collageOrder is None else collageOrder
			collageOutputDir = self.outputDir / collageDirStr
			collageOutputDir.mkdir(exist_ok=True)
			logger.info(f"Exporting collage to: {collageOutputDir}")

		assert startIx < endIx and startIx >= 0
		for t in trange(startIx, endIx, desc="[VideoRepresentationsExtractor::doExport]"):
			finalOutputs = {}
			for name, representation in self.tsr.items():
				finalOutputs[name] = representation[t]
		
			if exportCollage:
				images = [self.representations[k].makeImage(finalOutputs[k]) for k in collageOrder]
				images = collage_fn(images, titles=collageOrder)
				outImagePath = collageOutputDir / f"{t}.png"
				image_write(images, outImagePath)

	def __str__(self) -> str:
		nameRepresentations = ", ".join([f"'{x.name}'" for x in self.representations.values()])
		Str = "[Video Representations Extractor]"
		Str += f"\n  - Video: {self.video}"
		Str += f"\n  - Output directory: {self.outputDir}"
		Str += f"\n  - Output resolution: {self.outputResolution}"
		Str += f"\n  - Representations: ({len(self.representations)}): {nameRepresentations}"
		return Str
