#!/bin/bash
set -ex
export CWD=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

# download imgur upload script
test -e $CWD/imgur.sh || wget https://raw.githubusercontent.com/tremby/imgur.sh/main/imgur.sh -O $CWD/imgur.sh
chmod +x $CWD/imgur.sh

# Note: no need to start from existing data, it's just faster for CI. We could start from test_video.mp4 and generate
# the seed representations that are pre-stored in data/
test -d $CWD/data || tar -xzvf $CWD/data.tar.gz -C $CWD
rm -f $CWD/data_ci
cp -r $CWD/data $CWD/data_ci

vre $CWD/data_ci/rgb/npz/ -o $CWD/data_ci --config_path $CWD/cfg.yaml \
    -I $CWD/../../../examples/semantic_mapper/semantic_mapper.py:get_new_semantic_mapped_tasks \
    --output_dir_exists_mode skip_computed --frames 5 8 22

vre_collage $CWD/data_ci/ --config_path $CWD/cfg.yaml \
    -I $CWD/../../../examples/semantic_mapper/semantic_mapper.py:get_new_semantic_mapped_tasks \
    -o $CWD/collage --overwrite --video --fps 1

out_img=$CWD/collage/$(ls $CWD/collage/ | grep png | shuf | head -n 1)
test -f $out_img || ( echo "Image $out_img not found"; kill $$ )
bash $CWD/imgur.sh $out_img
