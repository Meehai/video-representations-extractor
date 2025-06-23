# VRE streaming

This example will show us various ways to stream from VRE to various external tools.

**Note: make sure nothing is printed to stdout except the frame data in case of MPL=0!**

## Matplotlib

The easiest thing we can do is use matplotlib. In order to use matplotlib, we can specify the `MPL=1` env variable.
```bash
MPL=1 VRE_DEVICE=cuda ./vre_streaming.py VIDEO CONFIG.YAML
```

## FFplay

```bash
VRE_DEVICE=cuda ./vre_streaming.py VIDEO CONFIG.YAML | \
ffplay \
  -f rawvideo \
  -pixel_format rgb24 \
  -video_size 1280x360 \
  -framerate 30 -
```

## MPV/VLC (via ffmpeg+TCP stream)

```bash
VRE_DEVICE=cuda ./vre_streaming.py VIDEO CONFIG.YAML | \
ffmpeg \
  -f rawvideo \
  -pixel_format rgb24 \
  -video_size 1280x360 \
  -framerate 30 \
  -i - \
  -f mpegts \
  -c:v libx264 \
  -tune zerolatency \
  -preset ultrafast \
  -b:v 1000k \
  tcp://localhost:9999?listen
```
and open in the video player (i.e. vlc) the stream at: `tcp://localhost:9999`.

## HTML5 (via ffmpeg+HLS stream)

```bash
VRE_DEVICE=cuda ./vre_streaming.py VIDEO CONFIG.YAML | \
ffmpeg \
  -f rawvideo \
  -pixel_format rgb24 \
  -video_size 1280x360 \
  -framerate 30 \
  -i - \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -g 30 \
  -f hls \
  -hls_time 0.1 \
  -hls_list_size 100 \
  -hls_flags delete_segments \
  -hls_segment_filename "stream_%03d.ts" \
  playlist.m3u8
```
In other terminal, serve the files
```bash
python -m http.server 9999 --bind 0.0.0.0 --cgi
```
And access it at `localhost:9999` (works on chromium at least)

## Stream from webcam

VRE video supports (via `ffmpeg` ofc) to get data from a `/dev/videoX` device on Linux. Follow this [gist](https://gist.github.com/Meehai/ab22a452ece0cd70d2c0da683d7e0122) that I made to get a webcam (from a phone for example) as a device on linux. Then you can just ffplay (or any of the stuff above) and pass as `VIDEO` the standard input (via `-`). Be careful on the resolutions.

```bash
ffmpeg \
  -f v4l2 \
  -framerate 30 \
  -video_size 640x480 \
  -i /dev/video9 \
  -pix_fmt rgb24 \
  -f rawvideo - | \
VRE_DEVICE=cuda MPL=0 ./vre_streaming.py - CONFIG.YAML \
  --input_size 480 640 \
  --output_size 360 1280 | \
ffplay \
  -f rawvideo \
  -pixel_format rgb24 \
  -video_size 1280x360 \
  -framerate 30 \
  -i -
```
