# VRE Video

A python video reader that can read video frames using ffmpeg behind the scenes. Support for PIL/Numpy based directory of frames as well + in memory numpy frames for testing. This repository is used to drive advancements/optimizations and supporting more video formats (thanks to ffmpeg). We have a dummy [video player](examples/vre-video-player/) as well in the examples.

Dependencies:
- `ffmpeg` If you want to use the standard video reader/writer, you need ffmpeg installed in your path.

Install (pip):
```bash
pip install vre-video
```

Install (dev):
```bash
git clone https://gitlab.com/meehai/ffmpeg-video/
echo "$(pwd)/ffmpeg-video" >> ~/.bashrc
source ~/.bashrc
pip install -r ffmpeg-video/requirements.txt
```

Handle venv/conda/uv stuff on your own!

## Project structure

```
vre_video/                 # the package
├── vre_video.py           # VREVideo: the only public class (readers/writers behind it)
├── utils.py               # logger, image_read/image_write, image_add_text (PIL helpers)
├── readers/               # FrameReader base + 4 impls + build_frame_reader() dispatch
│   ├── frame_reader.py    #   FrameReader ABC (shape/fps/path, __getitem__, __len__)
│   ├── ffmpeg_frame_reader.py  #   FFmpegFrameReader + FFprobe (mp4/mkv/URLs/v4l2 probe)
│   ├── pil_frame_reader.py     #   dir of png/jpg
│   ├── numpy_frame_reader.py   #   dir of npy/npz, in-memory ndarray, list of ndarrays
│   └── fd_frame_reader.py      #   FdFrameReader: raw frames from stdin/pipe/socket (async worker thread)
└── writers/                # FrameWriter base + 3 impls + build_frame_writer() dispatch
    ├── frame_writer.py
    ├── ffmpeg_frame_writer.py
    ├── pil_frame_writer.py
    └── numpy_frame_writer.py
cli/vre_video_player.py    # tkinter player (installed via setup.py scripts=)
examples/
├── vre-video-player/      # the same dummy video player, self-contained
└── video-live-coding/     # live-coding demo (main.py + src.py)
test/
├── unit/readers/          # per-reader tests + build_frame_reader dispatch tests
├── unit/writers/
└── integration/           # end-to-end VREVideo read/write tests
setup.py                   # packaging; scripts=[cli/vre_video_player.py]
requirements.txt           # dev deps
.gitlab-ci.yml             # CI: pylint (excludes test/, examples/) + pytest
```

Usage:
```python
from vre_video import VREVideo
video = VREVideo("video.mp4")
frame = video[ix] # returns a numpy array
```

Supports 3 backends for both reading and writing: `numpy`, `Pillow` and `ffmpeg`. It will auto-detect based on input: if a directory is provided it'll try to guess (png/jpg/npz/npy etc.) assuming it's a dir of frames (1.npz, ..., N.npz). If it's a path with suffix (i.e. .mp4, .mkv etc.) it will use the ffmpeg-based variant. Same for writing.

### Support for youtube videos

Requires `youtube-dl` python package.

Usage:
```python
from vre_video import VREVideo
video = VREVideo("https://www.youtube.com/...")
frame = video[ix] # returns a numpy array
```

### Support for stdin (like ffplay)

Usage:
```python
from vre_video import VREVideo
video = VREVideo("-")
frame = video[ix] # returns a numpy array
```

Used in combination with [vre_video_player](examples/vre-video-player/) (i.e. reading from `/dev/videoXX` webcam).

# Projects using VRE Video

## Video Representations Extractor (VRE)

Used in [Video Representations Extractor](https://gitlab.com/video-representations-extractor/video-representations-extractor) (VRE) library to extract multiple modalities for multi-task learning (MTL) in machine learning from videos

## Drones I/O & Actions (drone-ioact)

Used in [drone-ioact](https://gitlab.com/meehai/drone-ioact) as a base layer for the video player which acts as a testing ground for the simplest possible drone simulator environment: video continuously playing with basic actions like pause/play or take screenshot.

