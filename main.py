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
    parser.add_argument("--skip", type=int, default=0, help="Debug method to skip first N frames")
    parser.add_argument("--N", type=int, help="Debug method to only dump first N frames (starting from --skip)")
    args = parser.parse_args()

    args.videoPath = fullPath(args.videoPath)
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
    validRepresentations = []
    for representation in args.cfg["representations"]:
        name, method, group = representation["name"], representation["method"], representation["group"]
        Dir = args.outputDir / name
        thisCfgFile = Dir / "cfg.yaml"
        if Dir.exists():
            print("[validateOutputDir] Directory '%s' (%s/%s) already exists." % (Dir, group, name))
            representationCfg = yaml.safe_load(open(thisCfgFile, "r"))
            assert representationCfg == representation, "Wrong cfg file. Loaded: %s. This: %s" % \
                (representationCfg, representation)
            N = len([x for x in Dir.glob("*.npz")])
            if N == args.N:
                print("[validateOutputDir] Files already computted. Can be skipped safely.")
                continue
            if N != 0:
                assert N == args.N, "Loaded %d npz files. Expected %d." % (N, args.N)
            else:
                print("[validateOutputDir] Empty dir. New representation.")
        else:
            print("[validateOutputDir] Directory '%s' (%s/%s) doesn't exists. New representation." % (Dir, group, name))

        Dir.mkdir(exist_ok=True)
        yaml.safe_dump(representation, open(thisCfgFile, "w"))
        validRepresentations.append(representation)
    return validRepresentations

def updateRepresentations(representations):
    # Some representations are not lists (like RGB, which is unique). We create a list in order to process all
    #  representations the same. For example, depthEstimation has many potential solutions and we can provide a list of
    #  depth estimation methods.
    # We end up with a list of representations. Should look like this:
    #  [
    #    {'name': 'rgb', 'method': 'rgb', 'group': 'rgb'},
    #    {'name': 'hsv', 'method': 'hsv', 'group': 'hsv'},
    #    {'name': 'edges1', 'method': 'dexined', 'group': 'edgeDetection'},
    #    {'name': 'depth1', 'method': 'jiaw', 'group': 'depthEstimation'},
    #    {'name': 'depth2', 'method': 'dpt', 'group': 'depthEstimation'}
    #  ]
    result = []
    for k, v in representations.items():
        if not "name" in v:
            assert isinstance(v, dict)
            for k2, v2 in v.items():
                assert "name" in v2
                assert not "method" in v2
                item = v2
                item["method"] = k2
                item["group"] = k
                result.append(item)
        else:
            assert not "method" in v
            item = v
            item["method"] = k
            item["group"] = k
            result.append(item)
    return result

def validateArgs(args):
    args.video = tryReadVideo(args.videoPath, vidLib="pims")
    args.N = len(args.video) if args.N is None else args.N
    args.cfg = yaml.safe_load(open(args.cfgPath, "r"))
    args.cfg["resolution"] = list(map(lambda x : float(x), args.cfg["resolution"].split(",")))
    args.cfg["representations"] = updateRepresentations(args.cfg["representations"])
    return args

def main():
    args = getArgs()
    args.cfg["validRepresentations"] = validateOutputDir(args)

    print(args.video)
    print("[main] Num representations: %d. Num frames to be exported: %d. Skipping first %d frames." % \
        (len(args.cfg["validRepresentations"]), args.N, args.skip))

    names = [item["name"] for item in args.cfg["validRepresentations"]]
    representations = [getRepresentation(item) for item in args.cfg["validRepresentations"]]
    for i in trange(args.skip, args.skip + args.N):
        # frame = np.array(args.video[i])
        outputs = [r(args.video, i) for r in representations]
        # imgs = [r.makeImage(x) for r, x in zip(representations, outputs)]
        resizedOutputs = [imgResize(x, height=args.cfg["resolution"][0], width=args.cfg["resolution"][1], \
            onlyUint8=False) for x in outputs]
        outPaths = ["%s/%s/%d.npz" % (args.outputDir, name, i-args.skip) for name in names]
        for path, output in zip(outPaths, resizedOutputs):
            np.savez_compressed(path, output)

if __name__ == "__main__":
    main()
