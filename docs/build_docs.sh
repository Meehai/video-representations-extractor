#!/usr/bin/env bash
# docs/build_docs.sh -- build the docs site with opdoc (pip install -e ".[docs]").
#   README.md is the home page, docs/source/<page>/PAGE.md are the prose pages, vre/ and
#   vre_repository/ are the API. Vendored model code (*_impl, ultralytics, resources) is excluded.
#   Links are relative to the file holding them (as on GitLab); a broken one fails the build.
#
#   bash docs/build_docs.sh            # build into ./public, then print a file:// link
#   bash docs/build_docs.sh public     # same; GitLab Pages serves public/ in CI
set -euo pipefail
cd "$(dirname "$0")/.."                 # repo root (this script lives in docs/)
opdoc docs/ "${1:-public}" \
  --api-package vre --api-package vre_repository \
  --import-path . --import-path pkg/image_utils --import-path pkg/vre-video --import-path pkg/overrides \
  --exclude '*_impl' --exclude '*ultralytics' --exclude '*.resources' \
  --site-title "VRE documentation"
