#!/bin/bash
set -ex
export CWD=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
export VRE_ROOT=$CWD/../../../
export VID=$CWD/test_video.mp4

test -e $VID || \
    curl "https://gitlab.com/video-representations-extractor/video-representations-extractor/-/raw/master/resources/test_video.mp4" -o $VID

# download imgur upload script
test -e $CWD/imgur.sh || wget https://raw.githubusercontent.com/tremby/imgur.sh/main/imgur.sh -O $CWD/imgur.sh
chmod +x $CWD/imgur.sh

# make sure we start from scratch
test -f $CWD/cfg.yaml || ( echo "$CWD/cfg.yaml does not exist"; kill $$ )
n_frames=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 $VID)
X=$(shuf -i 1-"$n_frames" -n 1)
rm -rf $CWD/test_imgur/

# run VRE
vre $VID --cfg_path $CWD/cfg.yaml -o $CWD/test_imgur/ --start_frame $X --end_frame $((X+1)) --output_size 400 600
test "$?" -eq 0 || ( echo "VRE failed"; kill $$ )
vre_collage $CWD/test_imgur/ --overwrite

out_img=$CWD/test_imgur/collage/$X.png
test -f $out_img || ( echo "Image $out_img not found"; kill $$ )
# xdg-open $out_img
bash $CWD/imgur.sh $out_img
