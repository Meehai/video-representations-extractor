#!/bin/bash
set -ex

export CWD=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
export VRE_ROOT=$CWD/../../../../
export VID=$VRE_ROOT/resources/test_video.mp4

rm -f $CWD/frame.png
ffmpeg -i $VID -vframes 200 -f rawvideo -pix_fmt rgb24 - | \
  vre_streaming - $CWD/cfg_rgb.yaml --input_size 720 1280 --output_size 360 640 | \
  ffmpeg -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -i - -frames:v 1 $CWD/frame.png

[[ -f $CWD/frame.png && $(head -c 4 $CWD/frame.png) == $'\x89PNG' ]] \
  && echo "Valid PNG" || echo "Invalid or missing PNG"
