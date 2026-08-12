# Video Live Coding Editor

Run with (note: the `-` at the end is important):
```bash
./main.py test_video.mp4 | vlc --demux rawvideo --rawvid-fps 29.97 --rawvid-width 1280 --rawvid-height 720 --rawvid-chroma RV24 -
```

or

```bash
./main.py test_video.mp4 | vre_video_player.py - --input_size 720 1280
```

Live edit the `src.py` file such that it returns a new edited frame of the same shape.
