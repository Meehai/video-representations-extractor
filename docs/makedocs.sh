#!/usr/bin/bash
export CWD=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

rm -rf ${CWD}/source/docstring
sphinx-apidoc ${CWD}/../vre -o ${CWD}/source/docstring

cd ${CWD}
make clean
make html
cd --
