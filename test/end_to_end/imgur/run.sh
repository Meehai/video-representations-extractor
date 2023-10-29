#!/bin/bash

export FILE_ABS_PATH=$(readlink -f ${BASH_SOURCE[0]})
export CWD=$(dirname $FILE_ABS_PATH)

test -e imgur.sh || wget https://raw.githubusercontent.com/tremby/imgur.sh/main/imgur.sh -O imgur.sh # imgur upload script
chmod +x imgur.sh
test -e DJI_0956_720p.MP4 || gdown --no-cookies https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk # video
mkdir -p $VRE_WEIGHTS_DIR
cd $VRE_WEIGHTS_DIR
test -e  DJI_0956_velocities.npz || gdown https://drive.google.com/uc?id=1yafmmiHGMsgX6ym9Pbx2X2OxJtCY1xrT # velocities
test -e safeuav_semantic_0956_pytorch.ckpt || gdown https://drive.google.com/uc?id=1Up0pU1PRW0lzOzTEmV-c9eTLTe02kIpe  # semantic net pytorch

cd $CWD
test -f cfg.yaml || ( echo "cfg.yaml does not exist"; kill $$ ) # cfg
# TODO: CLI in this case is better than cfg
X=$(shuf -i 1-9000 -n 1)
export VRE_START_IX=$X
export VRE_END_IX=$((X+1))
export MPL_VIDEO_READ_BACKEND=pims
export MPL_IMAGE_BACKEND=opencv
echo $PATH
rm -rf test_imgur/

# run VRE
vre DJI_0956_720p.MP4 --cfg_path cfg.yaml -o test_imgur/
test "$?" -eq 0 || ( echo "VRE failed"; kill $$ )
vre_collage  test_imgur/ --no_video --overwrite

out_img=test_imgur/collage/$X.png
test -f $out_img || ( echo "Image $out_img not found"; kill $$ )
# xdg-open $out_img
test `which convert` && convert -resize 50% $out_img $out_img
bash imgur.sh $out_img
