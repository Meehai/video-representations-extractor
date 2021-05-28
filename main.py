import sys
import yaml
import numpy as np
from argparse import ArgumentParser
from tqdm import trange
from collections import OrderedDict
from media_processing_lib.video import tryReadVideo
from media_processing_lib.image import imgResize, tryWriteImage
from nwdata.utils import fullPath, topologicalSort

from representations import getRepresentation

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("--videoPath", required=True, help="Path to the scene video we are processing")
	parser.add_argument("--cfgPath", required=True, help="Path to global YAML cfg file")
	parser.add_argument("--outputDir", required=True, \
		help="Path to the output directory where representations are stored")
	parser.add_argument("--skip", type=int, default=0, help="Debug method to skip first N frames")
	parser.add_argument("--N", type=int, help="Debug method to only dump first N frames (starting from --skip)")
	parser.add_argument("--exportCollage", type=int, default=0, help="Export a stack of images as well? Default: 0.")
	args = parser.parse_args()

	args.videoPath = fullPath(args.videoPath)
	args.cfgPath = fullPath(args.cfgPath)
	args.outputDir = fullPath(args.outputDir)
	args.exportCollage = bool(args.exportCollage)
	args = validateArgs(args)
	return args

# Function that validates the output dir (args.outputDir) and each representation. By default, it just dumps the yaml
#  file of the representation under outputDir/representationName (i.e. video1/rgb/cfg.yaml). However, if the directory
#  outputDir/representationName exists, we check that the cfg file is identical to the gloal cfg file (args.cfgPath)
#  and we check that _all_ npz files have been computted. If so, we can safely skip this representation to avoid doing
#  duplicate work. If either the number of npz files is wrong or the cfg file is different, we throw and error and the
#  user must change the global cfg file for that particular representation: either by changing the resulting name or
#  updating the representation's parameters accordingly.
def makeOutputDirs(args):
	args.outputDir.mkdir(parents=True, exist_ok=True)
	if args.exportCollage:
		(args.outputDir / "collage").mkdir(exist_ok=True)
	for name, values in args.cfg["representations"].items():
		Dir = args.outputDir / name
		thisCfgFile = Dir / "cfg.yaml"
		if Dir.exists():
			loadedCfg = yaml.safe_load(open(thisCfgFile, "r"))
			assert loadedCfg == values, "Wrong cfg file. Loaded: %s. This: %s" % (loadedCfg, values)
			N = len([x for x in Dir.glob("*.npz")])
		else:
			Dir.mkdir(exist_ok=True)
			yaml.safe_dump(values, open(thisCfgFile, "w"))

def topoSortRepresentations(representations):
	result = OrderedDict()
	depGraph = {k : representations[k]["dependencies"] for k in representations}
	topoSort = topologicalSort(depGraph)
	for key in topoSort:
		result[key] = representations[key]
	return result

def validateArgs(args):
	args.video = tryReadVideo(args.videoPath, vidLib="pims")
	args.N = len(args.video) if args.N is None else args.N
	args.cfg = yaml.safe_load(open(args.cfgPath, "r"))
	args.cfg["resolution"] = list(map(lambda x : float(x), args.cfg["resolution"].split(",")))
	args.cfg["topoSortedRepresentations"] = topoSortRepresentations(args.cfg["representations"])
	return args

# Given a stack of N images, find the closest square X>=N*N and then remove rows 1 by 1 until it still fits X
# Example: 9: 3*3; 12 -> 4*4 -> 3*4 (3 rows). 65 => -> 9*9 -> 9*8
def makeCollage(images):
	N = len(images)
	x = int(np.sqrt(N))
	r, c = x, x
	# There are only 2 rows possible between x^2 and (x+1)^2 becuae (x+1)^2 = x^2 + 2*x + 1, thus we can add 2 columns
	#  at most. If a 3rd column is needed, then closest lower bound is (x+1)^2 and we must use that.
	if (c + 1) * r <= N:
		c += 1
	if (c + 1) * r <= N:
		c += 1
	assert (c + 1) * r > N
	images = np.array(images)
	images = images.reshape((r, c, *images.shape[1: ]))
	images = np.concatenate(np.concatenate(images, axis=1), axis=1)
	return images

def main():
	args = getArgs()
	makeOutputDirs(args)

	print(args.video)
	print("[main] Num representations: %d. Num frames to be exported: %d. Skipping first %d frames." % \
		(len(args.cfg["representations"]), args.N, args.skip))

	representations = {k : getRepresentation(v) for k, v in args.cfg["topoSortedRepresentations"].items()}
	height, width = args.cfg["resolution"]
	for t in trange(args.skip, args.skip + args.N):
		rawOutputs, finalOutputs, resizedImages = {}, {}, {}
		outPaths = {name : args.outputDir / name / ("%d.npz" % (t - args.skip)) for name in representations.keys()}
		outImagePath = args.outputDir / "collage" / ("%d.png" % (t-args.skip))

		# Topo sorted
		for name, representation in representations.items():
			depInputs = {k : finalOutputs[k] for k in args.cfg["representations"][name]["dependencies"]}
			outPath = outPaths[name]

			# Load if already computted
			if outPath.exists():
				resizedOutput = np.load(outPath)["arr_0"]
			# The current representation receives the current frame video[t] as well as the outputs of all
			#  dependencies of the current timestep.
			# TODO: Receive the representation itself and update __getattr__ to look into disk so we can access
			#  representation[t-k] (precomputted) or representation[t+k], not just representation[t]
			else:
				output = representation(args.video, t, depInputs)
				rawOutputs[name] = output
				resizedOutput = imgResize(output, height=height, width=width, onlyUint8=False)
				np.savez_compressed(outPath, resizedOutput)

			finalOutputs[name] = resizedOutput

		if args.exportCollage:
			notTopoSortedNames = list(args.cfg["representations"].keys())
			notTopoSortedRepresentations = [representations[k] for k in args.cfg["representations"].keys()]
			images = [r.makeImage(finalOutputs[k]) for k, r in zip(notTopoSortedNames, notTopoSortedRepresentations)]
			images = makeCollage(images)
			tryWriteImage(images, outImagePath)

if __name__ == "__main__":
	main()
