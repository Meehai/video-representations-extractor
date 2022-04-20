import yaml
from argparse import ArgumentParser
from media_processing_lib.video import video_read
from pathlib import Path
from video_representations_extractor import VideoRepresentationsExtractor

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("--videoPath", required=True, help="Path to the scene video we are processing")
	parser.add_argument("--cfgPath", required=True, help="Path to global YAML cfg file")
	parser.add_argument("--outputDir", required=True, \
		help="Path to the output directory where representations are stored")
	parser.add_argument("--outputResolution", required=True)
	parser.add_argument("--startFrame", type=int, default=0, help="The first frame. If not set, defaults to first.")
	parser.add_argument("--endFrame", type=int, help="End frame. If not set, defaults to length of video")
	parser.add_argument("--exportCollage", type=int, default=0, help="Export a stack of images as well? Default: 0.")
	args = parser.parse_args()

	args.videoPath = Path(args.videoPath).absolute()
	args.cfgPath = Path(args.cfgPath).absolute()
	args.outputDir = Path(args.outputDir).absolute()
	args.exportCollage = bool(args.exportCollage)
	args.outputResolution = list(map(lambda x : int(x), args.outputResolution.split(",")))
	args.cfg = yaml.safe_load(open(args.cfgPath, "r"))
	assert args.startFrame >= 0
	return args

def main():
	args = getArgs()
	video = video_read(args.videoPath, vid_lib="pims")

	vre = VideoRepresentationsExtractor(video, args.outputDir, representations=args.cfg, \
		outputResolution=args.outputResolution)
	endIx = len(video) if args.endFrame is None else args.endFrame
	vre.doExport(args.startFrame, endIx)

if __name__ == "__main__":
	main()
