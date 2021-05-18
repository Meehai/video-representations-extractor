import sys
import yaml
from argparse import ArgumentParser
from tqdm import trange
from media_processing_lib.video import tryReadVideo
from nwdata.utils import fullPath

from representations import getRepresentation

def getArgs():
    parser = ArgumentParser()
    parser.add_argument("--videoPath", required=True)
    parser.add_argument("--cfgPath", required=True)
    parser.add_argument("--outputDir", required=True)
    args = parser.parse_args()

    args.videoPath = fullPath(args.videoPath)
    args.cfgPath = fullPath(args.cfgPath)
    args.outputDir = fullPath(args.outputDir)
    args.cfg = yaml.safe_load(open(args.cfgPath, "r"))

    return args

def main():
    args = getArgs()
    assert len(args.cfg["representations"]) == 1 and "halftone" in args.cfg["representations"] and \
        args.cfg["representations"]["halftone"]["method"] == "python-halftone"

    video = tryReadVideo(args.videoPath, vidLib="pims")
    print(video)

    representations = [getRepresentation(r) for r in args.cfg["representations"]]
    print(representations)
    # N = len(video)
    N = 100
    for i in trange(N):
        frame = video[i]


if __name__ == "__main__":
    main()