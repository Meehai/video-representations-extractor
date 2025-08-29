# VRE streaming

This example will show us various ways to stream from VRE to various external tools.

## Matplotlib

The easiest thing we can do is use matplotlib. In order to use matplotlib, we can specify the `output_destination` flag to `matplotlib`.
```bash
VRE_DEVICE=cuda ./vre_streaming.py VIDEO CONFIG.YAML --output_destination matplotlib
```

By default `--output_destination` is set to stdout, so it must be combined with something that can understand raw video bytes.

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

VRE video supports (via `ffmpeg` ofc) to get data from a `/dev/videoX` device on Linux. Follow this [tutorial](./how2webcam.md) that I made to get a webcam (from a phone for example) as a device on linux. Then you can just ffplay (or any of the stuff above) and pass as `VIDEO` the standard input (via `-`). Be careful on the resolutions.

```bash
ffmpeg \
  -f v4l2 \
  -video_size 640x480 \
  -i /dev/video9 \
  -pix_fmt rgb24 \
  -f rawvideo - | \
VRE_DEVICE=cuda ./vre_streaming.py - CONFIG.YAML \
  --input_size 480 640 \
  --output_size 360 1280 | \
ffplay \
  -f rawvideo \
  -pixel_format rgb24 \
  -video_size 1280x360 \
  -framerate 30 \
  -i -
```

Note: If you are testing vre_streaming for an mp4 video and you don't want your frames to be skipped (like it'd happen for a continuous streaming session, like webcam), then also use `--disable_async_worker` that enables a sync (frame by frame) reader. For actual real-time streaming, keep it on, otherwise you will experience great lag.

## Stream from video but with vre_streaming on a remote server

You can combine the `VIDEO` and `--output_destination` parameters to act as a TCP socket (TODO for UDP):

### On your remote machine where you are processing the video frames

```bash
VRE_DEVICE=cuda ./vre_stremaing.py tcp://0.0.0.0:5000 CONFIG.YAML --output_destination socket
```

On your local machine (where your video or webcam) resides, you can do this:

```bash
ffmpeg \
  -f v4l2 \
  -video_size 640x480 \
  -i /dev/video9 \
  -pix_fmt rgb24 \
  -f rawvideo - | \
nc REMOTE_SERVER 5000 | \
ffplay \
  -f rawvideo \
  -pixel_format rgb24 \
  -video_size 1280x360 \
  -framerate 30 \
  -i -
```

#### You can only access your server via SSH (i.e. shared resources env) but it doesn't have a public IP

No problem, on your local machine start a SSH tunnel:

```bash
ssh -N -L 6000:localhost:5000 REMOTE_SERVER
```

And switch to `nc localhost 6000` instead of `nc REMOTE_SERVER 5000` which goes through the SSH tunnel.

#### Other stuff you can do
- ffmpeg -i video_path (instead of `v4l2`) with `nc` and `--output_destination socket`
- ffmpeg -i video_path and use `--output_destination stdout (default)` and pipe from the server to ffplay
  - the webcam is on device 1 which sends frame by frame to the processing server
  - the output is one the processing server
  - note: these can be on the same machine

Remember to use `--disable_async_worker` for mp4 files if you care about processing absolutely every frame.
