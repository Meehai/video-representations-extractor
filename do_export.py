import numpy as np
import yaml
from tqdm import trange
from functools import partial
from typing import Dict, Tuple
from pathlib import Path
from collections import OrderedDict
from nwdata.utils import topologicalSort, fullPath
from media_processing_lib.video import MPLVideo

from representations import getRepresentation
from video_representations_exporter import VideoRepresentationsExporter

def validateCfg(cfg):
	for name in cfg:
		representation = cfg[name]
		assert len(representation.keys()) == 3
		assert "method" in representation
		assert "dependencies" in representation
		assert "parameters" in representation

def topoSortRepresentations(representations):
	depGraph = {k : representations[k]["dependencies"] for k in representations}
	topoSort = topologicalSort(depGraph)
	return topoSort

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
		thisCfgFile = Dir / "cfg.yaml"
		values["outputResolution"] = outputResolution
		if Dir.exists():
			loadedCfg = yaml.safe_load(open(thisCfgFile, "r"))
			assert loadedCfg == values, "Wrong cfg file. Loaded: %s. This: %s" % (loadedCfg, values)
			N = len([x for x in Dir.glob("*.npz")])
		else:
			Dir.mkdir(exist_ok=True)
			yaml.safe_dump(values, open(thisCfgFile, "w"))

def doExport(video:MPLVideo, cfg:Dict, outputDir:Path, outputResolution:Tuple[int, int], \
	exportCollage:bool, N:int=None, skip:int=None):
	validateCfg(cfg)
	outputDir = fullPath(outputDir)
	makeOutputDirs(cfg, outputDir, outputResolution, exportCollage)

	skip = 0 if skip is None else skip
	N = len(video) - skip if N is None else N
	assert N > 0
	representations = cfg
	topoSortedRepresentations = topoSortRepresentations(representations)
	print(("[Video-Representations-Exporter::doExport] Video: %s. Num representations: %d." + \
		" Num frames to be exported: %d. Skipping first %d frames. Output dir: %s. Output resolution: %s") % (video, \
		len(representations), N, skip, outputDir, outputResolution))

	# Instantiating objects in correct oder
	tsr = OrderedDict()
	for name in topoSortedRepresentations:
		r = representations[name]
		dependencies = {k : tsr[k] for k in r["dependencies"]}
		objType = getRepresentation(r["method"])
		objType = partial(objType, baseDir=outputDir, name=name, dependencies=dependencies, \
			video=video, outShape=outputResolution)
		obj = objType(**r["parameters"]) if not r["parameters"] is None else objType()
		tsr[name] = obj

	startIx, endIx = skip, skip + N
	notTopoSortedNames = list(cfg.keys())
	vre = VideoRepresentationsExporter(tsr, exportCollage, outputDir / "collage", notTopoSortedNames)
	vre.doExport(startIx, endIx)
