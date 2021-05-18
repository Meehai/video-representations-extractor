import sys
from media_processing_lib.video import tryReadVideo

def main():
    video = tryReadVideo(sys.argv[1], vidLib="pims")
    print(video)

if __name__ == "__main__":
    main()