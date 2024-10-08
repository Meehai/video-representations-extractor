"""Mask2Former representation"""
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime
from typing import Any
from overrides import overrides
import torch as tr
from torch import nn
import numpy as np
from lovely_tensors import monkey_patch
from fvcore.common.config import CfgNode

from vre.representation import Representation, RepresentationOutput
from vre.logger import vre_logger as logger
from vre.utils import image_resize_batch, fetch_weights, image_read, image_write, load_weights
from vre.representations.semantic_segmentation.mask2former.mask2former_impl import MaskFormer as MaskFormerImpl
from vre.representations.semantic_segmentation.mask2former.mask2former_impl.det2_data.catalog \
    import MetadataCatalog, Metadata
from vre.representations.semantic_segmentation.mask2former.mask2former_impl.visualizer import Visualizer, ColorMode

monkey_patch()

def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int):
    """
    Compute the output size given input size and target short edge length.
    """
    scale = short_edge_length / min(oldh, oldw)
    newh, neww = (short_edge_length, scale * oldw) if oldh < oldw else (scale * oldh, short_edge_length)
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh, neww = newh * scale, neww * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    return newh, neww

class Mask2Former(Representation):
    """Mask2Former representation implementation. Note: only semantic segmentation (not panoptic/instance) enabled."""
    def __init__(self, model_id: str, semantic_argmax_only: bool, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(model_id, str) and model_id in {"47429163_0", "49189528_1", "49189528_0", "dummy"}, model_id
        self.semantic_argmax_only = semantic_argmax_only
        self.model_id = model_id
        self.model: MaskFormerImpl | None = None
        self.cfg: CfgNode | None = None
        self.metadata: Metadata | None = None

    @overrides(check_signature=False)
    def vre_setup(self, ckpt_data: dict | None = None):
        if self.model_id == "dummy":
            assert ckpt_data is not None
        else:
            assert ckpt_data is None
            weights_path = fetch_weights(__file__) / f"{self.model_id}.ckpt"
            assert isinstance(weights_path, Path), type(weights_path)
            ckpt_data = load_weights(weights_path)
        self.model, self.cfg, self.metadata = self._build_model(ckpt_data)
        self.model = self.model.to(self.device)

    @tr.no_grad()
    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
        height, width = frames.shape[1:3]
        _os = get_output_shape(height, width, self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        imgs = image_resize_batch(frames, _os[0], _os[1], "bilinear", "PIL").transpose(0, 3, 1, 2).astype("float32")
        inputs = [{"image": tr.from_numpy(img), "height": height, "width": width} for img in imgs]
        predictions: list[tr.Tensor] = [x["sem_seg"] for x in self.model(inputs)]
        res = []
        for pred in predictions:
            _pred = pred.argmax(0).byte() if self.semantic_argmax_only else pred.half().permute(1, 2, 0)
            res.append(_pred.to("cpu").numpy())
        return RepresentationOutput(output=np.stack(res))

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        res = []
        frames_rsz = image_resize_batch(frames, *repr_data.output.shape[1:3])
        for img, pred in zip(frames_rsz, repr_data.output):
            v = Visualizer(img, self.metadata, instance_mode=ColorMode.IMAGE_BW)
            _pred = pred if self.semantic_argmax_only else pred.argmax(-1)
            res.append(v.draw_sem_seg(_pred).get_image())
        res = np.stack(res)
        return res

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.output.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        interpolation = "nearest" if self.semantic_argmax_only else "bilinear"
        return RepresentationOutput(output=image_resize_batch(repr_data.output, *new_size, interpolation=interpolation))

    def _build_model(self, ckpt_data: dict[str, Any]) -> tuple[nn.Module, CfgNode, Metadata]:
        cfg = CfgNode(json.loads(ckpt_data["cfg"]))
        params = MaskFormerImpl.from_config(cfg)
        params = {**params, "semantic_on": True, "panoptic_on": False, "instance_on": False}
        model = MaskFormerImpl(**params).eval()
        res = model.load_state_dict(ckpt_data["state_dict"], strict=False) # inference only: we remove criterion
        assert res.unexpected_keys in (["criterion.empty_weight"], []), res
        model.to("cpu")
        assert len(cfg.DATASETS.TEST) == 1, cfg.DATASETS.TEST
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        return model, cfg, metadata

    def vre_free(self):
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()

def get_args() -> Namespace:
    """cli args"""
    parser = ArgumentParser()
    parser.add_argument("model_id", choices=["49189528_1", "47429163_0", "49189528_0"])
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--n_tries", type=int, default=1)
    return parser.parse_args()

def main(args: Namespace):
    """main fn. Usage: python mask2former.py 49189528_1/47429163_0/49189528_0 demo1.jpg output1.jpg"""
    img = image_read(args.input_image)

    m2f = Mask2Former(args.model_id_or_path, semantic_argmax_only=False, name="m2f", dependencies=[])
    m2f.device = "cuda" if tr.cuda.is_available() else "cpu"
    m2f.vre_setup()
    for _ in range(args.n_tries):
        now = datetime.now()
        pred = m2f.make(img[None])
        logger.info(f"Pred took: {datetime.now() - now}")
        semantic_result: np.ndarray = m2f.make_images(img[None], pred)[0]
        image_write(semantic_result, args.output_path)
    logger.info(f"Written prediction to '{args.output_path}'")

    # Sanity checks
    rtol = 1e-2
    if m2f.model_id == "47429163_0" and args.input_image.name == "demo1.jpg":
        assert np.allclose(mean := semantic_result.mean(), 129.41, rtol=rtol), (mean, semantic_result.std())
        assert np.allclose(std := semantic_result.std(), 53.33, rtol=rtol), std
    elif m2f.model_id == "49189528_1" and args.input_image.name == "demo1.jpg":
        assert np.allclose(mean := semantic_result.mean(), 125.23, rtol=rtol), (mean, semantic_result.std())
        assert np.allclose(std := semantic_result.std(), 48.89, rtol=rtol), std
    elif m2f.model_id == "49189528_0" and args.input_image.name == "demo1.jpg":
        assert np.allclose(mean := semantic_result.mean(), 118.47, rtol=rtol), (mean, semantic_result.std())
        assert np.allclose(std := semantic_result.std(), 52.08, rtol=rtol), std

if __name__ == "__main__":
    main(get_args())
