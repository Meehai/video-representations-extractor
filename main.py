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
    args.outputDir.mkdir(parents=True, exist_ok=False)
    [(args.outputDir / v["name"]).mkdir(parents=True) for k, v in args.cfg["representations"].items()]

    return args

def main():
    args = getArgs()

    video = tryReadVideo(args.videoPath, vidLib="pims")
    print(video)

    names = [v["name"] for k, v in args.cfg["representations"].items()]
    representations = [getRepresentation(k, v) for k, v in args.cfg["representations"].items()]
    N = len(video)
    for i in trange(len(video)):
        frame = np.array(video[i])
        frame = imgResize(frame, height=args.cfg["resolution"][0], width=args.cfg["resolution"][1])
        outputs = [r(frame) for r in representations]
        pngs = [r.makeImage(o) for r, o in zip(representations, outputs)]
        outPaths = ["%s/%s/%d.npz" % (args.outputDir, name, i) for name in names]
        for path, output, png in zip(outPaths, outputs, pngs):
            np.save(path, output)
            tryWriteImage(png, path.replace("npz", "png"))

if __name__ == "__main__":
    main()