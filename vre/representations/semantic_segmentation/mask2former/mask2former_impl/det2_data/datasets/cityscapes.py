# pylint: disable=all
# Copyright (c) Facebook, Inc. and its affiliates.
import logging

logger = logging.getLogger(__name__)

def load_cityscapes_instances(image_dir, gt_dir, from_json=True, to_polygons=True):
    raise NotImplementedError

def load_cityscapes_semantic(image_dir, gt_dir):
    raise NotImplementedError
