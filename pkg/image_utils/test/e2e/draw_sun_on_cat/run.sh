#!/bin/bash
set -ex
export CWD=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
export ROOT=$CWD/../../..

rm -f $CWD/res.jpg

$CWD/main.py $ROOT/resources/cat.png $ROOT/resources/sun.png $CWD/res.jpg

score=$($CWD/image_compare.py $ROOT/resources/combined.jpg $CWD/res.jpg)

echo "Score: $score"

if (( $(echo "$score >= 0.9955" | bc -l) )); then
    echo "OK"
    exit 0
else
    echo "not equal"
    exit 1
fi
