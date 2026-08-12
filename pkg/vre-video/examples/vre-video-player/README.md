# VRE Video Player

The best video player out there. Is your video 30FPS? Well, we can probably do 10FPS at most right now. Does it have
audio ? Well, we can't have audio. Yet.

Usage:
```bash
./vre_video_player.py /path/to/video
```

## read from stdin

```bash
command_that_output_bytes | ./vre_video_player.py - --resolution H W
```
