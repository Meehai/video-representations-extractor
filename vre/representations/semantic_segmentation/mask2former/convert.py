"""Mask2former conversion script. See README.md on how to use."""
# pylint: disable=all
import torch as tr
import json
from pathlib import Path
import sys
sys.path.append(f"{Path(__file__).parent / 'detectron2'}") # only needed if you can't install detectron but can clone it

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config, MaskFormer # official mask2former repo installed from github, not in vre

CFG = sys.argv[1]
WEIGHTS = sys.argv[2]
OUTPUT_WEIGHTS = sys.argv[3]

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file(CFG)
cfg.MODEL.WEIGHTS = WEIGHTS
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
if not tr.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"

model = MaskFormer(cfg)
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

out_data = {"state_dict": model.state_dict(), "cfg": json.dumps(cfg)}
tr.save(out_data, OUTPUT_WEIGHTS)
print(f"saved converted weigts & cfg at: '{OUTPUT_WEIGHTS}'")
