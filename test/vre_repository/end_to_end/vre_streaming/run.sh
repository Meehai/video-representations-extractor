#!/bin/bash
set -ex
export CWD=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
export VRE_ROOT=$CWD/../../../
export VID=$CWD/test_video.mp4

ffmpeg -i $VID -vframes 100 -f rawvideo -pix_fmt rgb24 - | \
  vre_streaming - $CWD/cfg_rgb.yaml --input_size 720 1280 --output_size 360 1280 | \
  ffplay -f rawvideo -pixel_format rgb24 -video_size 1280x360 -framerate 30 -autoexit -nodisp -
