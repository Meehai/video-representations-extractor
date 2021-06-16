import yaml
from argparse import ArgumentParser
from media_processing_lib.video import tryReadVideo
from nwdata.utils import fullPath

from do_export import *

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("--videoPath", required=True, help="Path to the scene video we are processing")
	parser.add_argument("--cfgPath", required=True, help="Path to global YAML cfg file")
	parser.add_argument("--outputDir", required=True, \
		help="Path to the output directory where representations are stored")
	parser.add_argument("--outputResolution", required=True)
	parser.add_argument("--skip", type=int, default=0, help="Debug method to skip first N frames")
	parser.add_argument("--N", type=int, help="Debug method to only dump first N frames (starting from --skip)")
	parser.add_argument("--exportCollage", type=int, default=0, help="Export a stack of images as well? Default: 0.")
	args = parser.parse_args()

	args.videoPath = fullPath(args.videoPath)
	args.cfgPath = fullPath(args.cfgPath)
	args.outputDir = fullPath(args.outputDir)
	args.exportCollage = bool(args.exportCollage)
	args.outputResolution = list(map(lambda x : int(x), args.outputResolution.split(",")))
	args.cfg = yaml.safe_load(open(args.cfgPath, "r"))
	return args

def main():
	args = getArgs()
	video = tryReadVideo(args.videoPath, vidLib="pims")

	# validateCfg(args.cfg)
	# makeOutputDirs(args.cfg, args.outputDir, args.outputResolution, args.exportCollage)

	representations = args.cfg
	vre = VideoRepresentationsExtractor(video, args.outputDir, args.outputResolution, representations, False)
	endIx = len(video) - args.skip if args.N is None else min(len(video) - args.skip, args.N)
	vre.doExport(args.skip, endIx)

	# doExport(video, args.cfg, args.outputDir, args.outputResolution, args.exportCollage, args.N, args.skip)

if __name__ == "__main__":
	main()
