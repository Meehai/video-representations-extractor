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
    parser.add_argument("--videoPath", required=True, help="Path to the scene video we are processing")
    parser.add_argument("--cfgPath", required=True, help="Path to global YAML cfg file")
    parser.add_argument("--outputDir", required=True, \
        help="Path to the output directory where representations are stored")
    parser.add_argument("--N", type=int, help="Debug method to only dump first N frames")
    args = parser.parse_args()

    args.videoPath = args.videoPath
    args.cfgPath = fullPath(args.cfgPath)
    args.outputDir = fullPath(args.outputDir)
    args = validateArgs(args)
    return args

# Function that validates the output dir (args.outputDir) and each representation. By default, it just dumps the yaml
#  file of the representation under outputDir/representationName (i.e. video1/rgb/cfg.yaml). However, if the directory
#  outputDir/representationName exists, we check that the cfg file is identical to the gloal cfg file (args.cfgPath)
#  and we check that _all_ npz files have been computted. If so, we can safely skip this representation to avoid doing
#  duplicate work. If either the number of npz files is wrong or the cfg file is different, we throw and error and the
#  user must change the global cfg file for that particular representation: either by changing the resulting name or
#  updating the representation's parameters accordingly.
def validateOutputDir(args):
    args.outputDir.mkdir(parents=True, exist_ok=True)
    validRepresentations = {}
    for k, v in args.cfg["representations"].items():
        Dir = args.outputDir / v["name"]
        thisCfgFile = Dir / "cfg.yaml"
        if Dir.exists():
            print("[validateOutputDir] Directory '%s' (%s) already exists." % (Dir, k))
            representationCfg = yaml.safe_load(open(thisCfgFile, "r"))
            assert representationCfg == v, "Wrong cfg file. Loaded: %s. This: %s" % (representationCfg, v)
            N = len([x for x in Dir.glob("*.npz")])
            if N == args.N:
                print("[validateOutputDir] Files already computted. Can be skipped safely.")
                continue
            if N != 0:
                assert N == args.N, "Loaded %d npz files. Expected %d." % (N, args.N)
            else:
                print("[validateOutputDir] Empty dir. New representation.")
        else:
            print("[validateOutputDir] Directory '%s' (%s) doesn't exists. New representation." % (Dir, k))

        Dir.mkdir(exist_ok=True)
        yaml.safe_dump(v, open(thisCfgFile, "w"))
        validRepresentations[k] = v
    return validRepresentations

def validateArgs(args):
    args.video = tryReadVideo(args.videoPath, vidLib="pims")
    args.N = len(args.video) if args.N is None else args.N
    args.cfg = yaml.safe_load(open(args.cfgPath, "r"))
    args.cfg["resolution"] = list(map(lambda x : float(x), args.cfg["resolution"].split(",")))
    for k, v in args.cfg["representations"].items():
        v["method"] = k if not "method" in v else v["method"]
    return args

def main():
    args = getArgs()
    args.cfg["validRepresentations"] = validateOutputDir(args)

    print(args.video)
    print("[main] Num representations: %d. Num frames to be exported: %d" % \
        (len(args.cfg["validRepresentations"]), args.N))

    names = [v["name"] for k, v in args.cfg["validRepresentations"].items()]
    representations = [getRepresentation(k, v) for k, v in args.cfg["validRepresentations"].items()]
    for i in trange(args.N):
        frame = np.array(args.video[i])
        frame = imgResize(frame, height=args.cfg["resolution"][0], width=args.cfg["resolution"][1])
        outputs = [r(frame) for r in representations]
        outPaths = ["%s/%s/%d.npz" % (args.outputDir, name, i) for name in names]
        for path, output in zip(outPaths, outputs):
            np.savez_compressed(path, output)

if __name__ == "__main__":
    main()
