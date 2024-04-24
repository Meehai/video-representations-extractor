#!/bin/bash

export FILE_ABS_PATH=$(readlink -f ${BASH_SOURCE[0]})
export CWD=$(dirname $FILE_ABS_PATH)
old_cwd=$(pwd)


# download stuff
## imgur upload script
test -e $CWD/imgur.sh || wget https://raw.githubusercontent.com/tremby/imgur.sh/main/imgur.sh -O $CWD/imgur.sh
chmod +x $CWD/imgur.sh
## test video
test -e $CWD/DJI_0956_720p.MP4 \
    || gdown --no-cookies https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk \
        -O $CWD/DJI_0956_720p.MP4
## download weights
mkdir -p $VRE_WEIGHTS_DIR
### velocities for odo flow
test -e $VRE_WEIGHTS_DIR/DJI_0956_velocities.npz || \
    gdown https://drive.google.com/uc?id=1yafmmiHGMsgX6ym9Pbx2X2OxJtCY1xrT -O $VRE_WEIGHTS_DIR/DJI_0956_velocities.npz
### weights for safeuav semantic
test -e $VRE_WEIGHTS_DIR/safeuav_semantic_0956_pytorch.ckpt || \
    gdown https://drive.google.com/uc?id=1Up0pU1PRW0lzOzTEmV-c9eTLTe02kIpe \
        -O $VRE_WEIGHTS_DIR/safeuav_semantic_0956_pytorch.ckpt

# make sure we start from scratch
cd $CWD;
test -f cfg.yaml || ( echo "cfg.yaml does not exist"; kill $$ )
X=$(shuf -i 1-9000 -n 1)
rm -rf test_imgur/

# run VRE
vre DJI_0956_720p.MP4 --cfg_path cfg.yaml -o test_imgur/ --start_frame $X --end_frame $((X+1))
test "$?" -eq 0 || ( echo "VRE failed"; kill $$ )
vre_collage test_imgur/ --overwrite

out_img=test_imgur/collage/$X.png
test -f $out_img || ( echo "Image $out_img not found"; kill $$ )
# xdg-open $out_img
test `which convert` && convert -resize 50% $out_img $out_img
bash imgur.sh $out_img

# go back to where we were
cd $old_cwd
