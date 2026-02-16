#!/bin/bash
set -ex
export CWD=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
export VRE_ROOT=$CWD/../../../
export VID=$CWD/test_video.mp4

test -e $VID || curl "https://gitlab.com/video-representations-extractor/video-representations-extractor/-/raw/master/resources/test_video.mp4" -o $VID

# download imgur upload script
test -e $CWD/imgur.sh || wget https://raw.githubusercontent.com/tremby/imgur.sh/main/imgur.sh -O $CWD/imgur.sh
chmod +x $CWD/imgur.sh

# make sure we start from scratch
test -f $CWD/cfg.yaml || { echo "$CWD/cfg.yaml does not exist"; exit 1; }
n_frames=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 $VID)
X=$(shuf -i 1-"$n_frames" -n 1)
# rm -rf $CWD/test_imgur/

# run vre, vre_reader & vre_collage
vre $VID --config_path $CWD/cfg.yaml -o $CWD/test_imgur/ --frames $X --output_dir_exists_mode skip_computed
vre_reader $CWD/test_imgur/ --config_path $CWD/cfg.yaml
vre_collage $CWD/test_imgur/ --config_path $CWD/cfg.yaml -o $CWD/collage --overwrite

out_img=$CWD/collage/$X.png
test -f $out_img || { echo "Image $out_img not found"; exit 1; }
bash $CWD/imgur.sh $out_img || { echo "Could not upload to imgur"; }
