#!/usr/bin/bash

set -e
export CWD=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
export TYPEGUARD=1

function run {
    echo "==========================================================================="
    echo "Running '$1'"
    bash $CWD/$1/run.sh
    echo "==========================================================================="
}

run draw_sun_on_cat
