import sys
import yaml
import numpy as np
from argparse import ArgumentParser
from tqdm import trange
from media_processing_lib.video import tryReadVideo
from media_processing_lib.image import imgResize, tryWriteImage
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
    args.cfg["resolution"] = list(map(lambda x : float(x), args.cfg["resolution"].split(",")))

    assert not args.outputDir.exists()

    return args

def main():
    args = getArgs()
    assert len(args.cfg["representations"]) == 1 and "halftone" in args.cfg["representations"] and \
        args.cfg["representations"]["halftone"]["method"] == "python-halftone"

    video = tryReadVideo(args.videoPath, vidLib="pims")
    print(video)

    names = [v["name"] for k, v in args.cfg["representations"].items()]
    representations = [getRepresentation(k, v) for k, v in args.cfg["representations"].items()]
    N = len(video)
    N = 100
    args.outputDir.mkdir(parents=True, exist_ok=False)
    [(args.outputDir / v["name"]).mkdir(parents=True) for k, v in args.cfg["representations"].items()]
    for i in trange(N):
        frame = np.array(video[i])
        frame = imgResize(frame, height=args.cfg["resolution"][0], width=args.cfg["resolution"][1])
        outputs = [r(frame) for r in representations]
        outPaths = ["%s/%s/%d.npz" % (args.outputDir, name, i) for name in names]
        for path, output in zip(outPaths, outputs):
            np.save(path, output)
            tryWriteImage(np.uint8(output * 255), path.replace("npz", "png"))

if __name__ == "__main__":
    main()