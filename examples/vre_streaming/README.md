# VRE streaming

This example will show us various ways to stream from VRE to various external tools.

**Note: make sure nothing is printed to stdout except the frame data in case of MPL=0!**

## Matplotlib

The easiest thing we can do is use matplotlib. In order to use matplotlib, we can specify the `MPL=1` env variable.
```bash
MPL=1 ./vre_streaming.py | ffplay -f rawvideo -pixel_format rgb24 -video_size 1280x360 -framerate 30 -
```

## FFplay

```bash
MPL=0 VRE_LOGLEVEL=-1 VRE_PBAR=0 ./vre_streaming.py | ffplay -f rawvideo -pixel_format rgb24 -video_size 1280x360 -framerate 30 -
```

## MPV/VLC (via ffmpeg+TCP stream)

```bash
MPL=0 VRE_LOGLEVEL=-1 VRE_PBAR=0 ./vre_streaming.py | ffmpeg \
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
MPL=0 VRE_LOGLEVEL=-1 VRE_PBAR=0 ./vre_streaming.py | ffmpeg \
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
  -hls_time 2 \
  -hls_list_size 5 \
  -hls_flags delete_segments \
  -hls_segment_filename "stream_%03d.ts" \
  playlist.m3u8
```
In other terminal, serve the files
```bash
python -m http.server 9999 --bind 0.0.0.0 --cgi
```
And access it at `localhost:9999` (works on chromium at least)
