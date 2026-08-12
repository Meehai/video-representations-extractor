# Change resolution based on v4l2 driver

**Created**: 2025-06-21
**Closed**: 2025-06-23
**Priority**: 2

ffplay does this:

```
ffplay -f v4l2 -video_size 800x480 /dev/video9
[video4linux2,v4l2 @ 0x7414d8000c80] The V4L2 driver changed the video from 800x480 to 640x480
```

Upstream: video-representations-extractor/vre-video#2 (https://gitlab.com/video-representations-extractor/vre-video/-/work_items/2)
