from main import getArgs, tryReadVideo, getRepresentation, np, trange, tryWriteImage

def suffix(x:int):
    x = str(x)
    if not x[-1] in ("1", "2", "3") or (len(x) > 1 and x[-2] == "1"):
        return "th"
    elif x[-1] == "1":
        return "st"
    elif x[-1] == "2":
        return "nd"
    else:
        return "rd"

def main():
    args = getArgs()
    assert args.outputDir.exists()
    video = tryReadVideo(args.videoPath, vidLib="pims")

    names = [item["name"] for item in args.cfg["representations"]]
    representations = [getRepresentation(item) for item in args.cfg["representations"]]
    (args.outputDir / "collage").mkdir(exist_ok=True)
    for i in trange(len(video)):
        inPaths = ["%s/%s/%d.npz" % (args.outputDir, name, i) for name in names]
        try:
            outputs = [np.load(x)["arr_0"] for x in inPaths]
        except Exception as e:
            print("[main_make_collage] Error loading %d%s npz file. Ending prematurely." % (i + 1, suffix(i + 1)))
            break
        pngs = [r.makeImage(o) for r, o in zip(representations, outputs)]
        png = np.concatenate(pngs, axis=1)
        tryWriteImage(png, "%s/collage/%d.png" % (args.outputDir, i))

if __name__ == "__main__":
    main()