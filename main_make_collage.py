from main import getArgs, tryReadVideo, getRepresentation, np, trange, tryWriteImage

def main():
    args = getArgs()
    assert args.outputDir.exists()
    video = tryReadVideo(args.videoPath, vidLib="pims")

    names = [v["name"] for k, v in args.cfg["representations"].items()]
    representations = [getRepresentation(k, v) for k, v in args.cfg["representations"].items()]
    (args.outputDir / "collage").mkdir(exist_ok=True)
    for i in trange(len(video)):
        inPaths = ["%s/%s/%d.npz" % (args.outputDir, name, i) for name in names]
        outputs = [np.load(x)["arr_0"] for x in inPaths]
        pngs = [r.makeImage(o) for r, o in zip(representations, outputs)]
        png = np.concatenate(pngs, axis=1)
        tryWriteImage(png, "%s/collage/%d.png" % (args.outputDir, i))

if __name__ == "__main__":
    main()