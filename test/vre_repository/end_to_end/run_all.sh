#!/usr/bin/bash

export CWD=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
export PYTHONPATH="$PYTHONPATH:$CWD/../../../"
export PATH="$PATH:$CWD/../../../cli"

bash $CWD/vre_batched/run.sh
bash $CWD/task_mapped_representations/run.sh
bash $CWD/vre_streaming/run.sh
#python $CWD/vre_batched/test_vre_batched.py
