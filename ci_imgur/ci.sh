#!/bin/bash

test -e DJI_0956_720p.MP4 || gdown https://drive.google.com/uc?id=158U-W-Gal6eXxYtS1ca1DAAxHvknqwAk # video
mkdir -p ../weights
cd ../weights
test -e  DJI_0956_velocities.npz || gdown https://drive.google.com/uc?id=1yafmmiHGMsgX6ym9Pbx2X2OxJtCY1xrT # velocities
test -e safeuav_semantic_0956.pkl || gdown https://drive.google.com/uc?id=1tgni1GnshQq_zCqjreBxHropRfKBQ2HH # semantic net pytorch
cd -
test -e cfg.yaml || wget https://pastebin.com/raw/D0awQLZP -O cfg.yaml # cfg
X=$(shuf -i 1-9000 -n 1)
python3 ../scripts/main.py --videoPath DJI_0956_720p.MP4 --cfgPath cfg.yaml --outputDir test_imgur/ --outputResolution 240,426 --startFrame $X --endFrame $((X+1))
convert -resize 50% test_imgur//collage/$X.png test_imgur//collage/$X.png
bash imgur.sh test_imgur//collage/$X.png
