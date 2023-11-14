# pylint: disable=all
import torch

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from .utils import bbox_iou

class FastSAMPredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'
