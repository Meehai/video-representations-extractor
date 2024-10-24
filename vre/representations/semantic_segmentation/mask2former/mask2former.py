"""Mask2Former representation"""
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from overrides import overrides
import torch as tr
import numpy as np
from lovely_tensors import monkey_patch

from vre.representations import Representation, ReprOut, LearnedRepresentationMixin
from vre.logger import vre_logger as logger
from vre.utils import (image_resize_batch, fetch_weights, image_read, image_write,
                       vre_load_weights, colorize_semantic_segmentation)

try:
    from .mask2former_impl import MaskFormer as MaskFormerImpl, CfgNode
except ImportError:
    from mask2former_impl import MaskFormer as MaskFormerImpl, CfgNode

monkey_patch()

def _get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int):
    """Compute the output size given input size and target short edge length."""
    scale = short_edge_length / min(oldh, oldw)
    newh, neww = (short_edge_length, scale * oldw) if oldh < oldw else (scale * oldh, short_edge_length)
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh, neww = newh * scale, neww * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    return newh, neww

class Mask2Former(Representation, LearnedRepresentationMixin):
    """Mask2Former representation implementation. Note: only semantic segmentation (not panoptic/instance) enabled."""
    def __init__(self, model_id: str, semantic_argmax_only: bool, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(model_id, str) and model_id in {"47429163_0", "49189528_1", "49189528_0"}, model_id
        self._m2f_resources = Path(__file__).parent / "mask2former_impl/resources"
        self.classes, self.color_map, self.thing_dataset_id_to_contiguous_id = self._get_metadata(model_id)
        self.semantic_argmax_only = semantic_argmax_only
        self.model_id = model_id
        self.model: MaskFormerImpl | None = None
        self.cfg: CfgNode | None = None

    def vre_setup(self, load_weights = True):
        assert self.model is None, "vre_setup already called before"
        weights_path = fetch_weights(__file__) / f"{self.model_id}.ckpt"
        assert isinstance(weights_path, Path), type(weights_path)
        ckpt_data = vre_load_weights(weights_path)
        self.cfg = CfgNode(json.load(open(f"{self._m2f_resources}/{self.model_id}_cfg.json", "r")))
        params = MaskFormerImpl.from_config(self.cfg)
        params["metadata"] = SimpleNamespace(thing_dataset_id_to_contiguous_id=self.thing_dataset_id_to_contiguous_id)
        self.model = MaskFormerImpl(**{**params, "semantic_on": True, "panoptic_on": False, "instance_on": False})
        if load_weights:
            res = self.model.load_state_dict(ckpt_data["state_dict"], strict=False) # inference only: remove criterion
            assert res.unexpected_keys in (["criterion.empty_weight"], []), res
        self.model = self.model.eval().to(self.device)

    @tr.no_grad()
    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, ReprOut] | None = None) -> ReprOut:
        height, width = frames.shape[1:3]
        _os = _get_output_shape(height, width, self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        imgs = image_resize_batch(frames, _os[0], _os[1], "bilinear", "PIL").transpose(0, 3, 1, 2).astype("float32")
        inputs = [{"image": tr.from_numpy(img), "height": height, "width": width} for img in imgs]
        predictions: list[tr.Tensor] = [x["sem_seg"] for x in self.model(inputs)]
        res = []
        for pred in predictions:
            _pred = pred.argmax(0).byte() if self.semantic_argmax_only else pred.half().permute(1, 2, 0)
            res.append(_pred.to("cpu").numpy())
        return ReprOut(output=np.stack(res))

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: ReprOut) -> np.ndarray:
        res = []
        frames_rsz = image_resize_batch(frames, *repr_data.output.shape[1:3])
        for img, pred in zip(frames_rsz, repr_data.output):
            _pred = pred if self.semantic_argmax_only else pred.argmax(-1)
            res.append(colorize_semantic_segmentation(_pred, self.classes, self.color_map, img))
        res = np.stack(res)
        return res

    @overrides
    def size(self, repr_data: ReprOut) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        interpolation = "nearest" if self.semantic_argmax_only else "bilinear"
        return ReprOut(output=image_resize_batch(repr_data.output, *new_size, interpolation=interpolation))

    def _get_metadata(self, model_id: str) -> tuple[list[str], list[tuple[int, int, int]], dict[str, int]]:
        metadata = None
        if model_id == "49189528_1":
            metadata = json.load(open(f"{self._m2f_resources}/mapillary_metadata.json", "r"))
        if model_id == "47429163_0":
            metadata = json.load(open(f"{self._m2f_resources}/coco_metadata.json", "r"))
        if model_id == "49189528_0":
            metadata = json.load(open(f"{self._m2f_resources}/mapillary_metadata2.json", "r"))
        return metadata["stuff_classes"], metadata["stuff_colors"], metadata.get("thing_dataset_id_to_contiguous_id")

    def vre_free(self):
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
        self.model = None

def get_args() -> Namespace:
    """cli args"""
    parser = ArgumentParser()
    parser.add_argument("model_id", choices=["49189528_1", "47429163_0", "49189528_0"])
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args()

def main(args: Namespace):
    """main fn. Usage: python mask2former.py 49189528_1/47429163_0/49189528_0 demo1.jpg output1.jpg"""
    img = image_read(args.input_image)

    m2f = Mask2Former(args.model_id, semantic_argmax_only=False, name="m2f", dependencies=[])
    m2f.device = "cuda" if tr.cuda.is_available() else "cpu"
    m2f.vre_setup()
    now = datetime.now()
    pred = m2f.make(img[None])
    logger.info(f"Pred took: {datetime.now() - now}")
    semantic_result: np.ndarray = m2f.make_images(img[None], pred)[0]
    image_write(semantic_result, args.output_path)
    logger.info(f"Written prediction to '{args.output_path}'")
    return semantic_result # for integration tests

if __name__ == "__main__":
    main(get_args())
