"""Mask2Former representation"""
import json
from pathlib import Path
import sys
from overrides import overrides
import torch as tr
from torch import nn
import numpy as np
from lovely_tensors import monkey_patch
from PIL import Image
from fvcore.common.config import CfgNode

monkey_patch()

try:
    from .mask2former_impl import MaskFormer as MaskFormerImpl
    from .mask2former_impl.det2_data import MetadataCatalog
    from .mask2former_impl.visualizer import Visualizer, ColorMode
    from ....representation import Representation, RepresentationOutput
    from ....logger import logger
    from ....utils import gdown_mkdir, image_resize_batch, VREVideo, get_weights_dir
except ImportError: # when running this script directly
    from mask2former_impl import MaskFormer as MaskFormerImpl
    from mask2former_impl.visualizer import Visualizer, ColorMode
    from mask2former_impl.det2_data import MetadataCatalog
    from vre.representation import Representation, RepresentationOutput
    from vre.logger import logger
    from vre.utils import gdown_mkdir, image_resize_batch, VREVideo, get_weights_dir

def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int):
    """
    Compute the output size given input size and target short edge length.
    """
    h, w = oldh, oldw
    scale = short_edge_length / min(h, w)
    newh, neww = (short_edge_length, scale * w) if h < w else (scale * h, short_edge_length)
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    return newh, neww

def apply_image(img: np.ndarray, h, w, new_h, new_w):
    """apply image transform -- see if we need the non uint8 variant"""
    assert img.shape[:2] == (h, w)
    assert len(img.shape) <= 4
    interp_method = Image.BILINEAR
    assert img.dtype == np.uint8, img.dtype

    if len(img.shape) > 2 and img.shape[2] == 1:
        pil_image = Image.fromarray(img[:, :, 0], mode="L")
    else:
        pil_image = Image.fromarray(img)
    pil_image = pil_image.resize((new_w, new_h), interp_method)
    ret = np.asarray(pil_image)
    if len(img.shape) > 2 and img.shape[2] == 1:
        ret = np.expand_dims(ret, -1)
    return ret

class Mask2Former(Representation):
    """Mask2Former representation implementation. Note: only semantic segmentation (not panoptic/instance) enabled."""
    def __init__(self, model_id: str, semantic_argmax_only: bool, **kwargs):
        super().__init__(**kwargs)
        weights_path = self._get_weights(model_id)
        self.model, self.cfg, self.metadata = self._build_model(weights_path)
        self.model_id = model_id
        self.device = "cpu"
        self.semantic_argmax_only = semantic_argmax_only

    # pylint: disable=arguments-differ
    @overrides(check_signature=False)
    def vre_setup(self, video: VREVideo, device: str):
        self.model = self.model.to(device)
        self.device = device

    @tr.no_grad()
    @overrides
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        height, width = frames.shape[1:3]
        _os = get_output_shape(height, width, self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        imgs = [apply_image(img, height, width, _os[0], _os[1]).astype("float32").transpose(2, 0, 1) for img in frames]
        inputs = [{"image": tr.from_numpy(img), "height": height, "width": width} for img in imgs]
        predictions = [x["sem_seg"] for x in self.model(inputs)]
        res = []
        for pred in predictions:
            _pred = pred.argmax(0).byte() if self.semantic_argmax_only else pred.half().permute(1, 2, 0)
            res.append(_pred.to("cpu").numpy())
        return np.stack(res)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        res = []
        frames_rsz = image_resize_batch(frames, *repr_data.shape[1:3])
        for img, pred in zip(frames_rsz, repr_data):
            v = Visualizer(img, self.metadata, instance_mode=ColorMode.IMAGE_BW)
            _pred = pred if self.semantic_argmax_only else pred.argmax(-1)
            res.append(v.draw_sem_seg(_pred).get_image())
        res = np.stack(res)
        return res

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        interpolation = "nearest" if self.semantic_argmax_only else "bilinear"
        return image_resize_batch(repr_data, *new_size, interpolation=interpolation)

    def _get_weights(self, model_id: str | dict) -> str:
        links = {
            "47429163_0": "https://drive.google.com/u/0/uc?id=1a5WOek1NyEqccBJuZQKSgsUU7drzjLY5", # COCO SWIN
            "49189528_1": "https://drive.google.com/u/0/uc?id=1Ypzs7nXoqxsYwLrlt2rojr6T9t2MJkLH", # Mapillary R50
            "49189528_0": "https://drive.google.com/u/0/uc?id=1fQevKfynhTqYI-7qinQp9ewQbDMT6NTN" # Mapillary SWIN
        }
        if isinstance(model_id, dict):
            logger.warning("Unknown model provided as dict. Loading as is.")
            return model_id
        if model_id not in links.keys():
            logger.warning(f"Unknown model provided: {model_id}. Loading as is.")
            return model_id

        weights_path = get_weights_dir() / f"{model_id}.ckpt"
        if not weights_path.exists():
            gdown_mkdir(links[model_id], weights_path)
        return weights_path

    def _build_model(self, weights_path: Path | dict) -> tuple[nn.Module, CfgNode, MetadataCatalog]:
        ckpt_data = tr.load(weights_path, map_location="cpu") if isinstance(weights_path, Path) else weights_path
        cfg = CfgNode(json.loads(ckpt_data["cfg"]))
        params = MaskFormerImpl.from_config(cfg)
        params = {**params, "semantic_on": True, "panoptic_on": False, "instance_on": False}
        model = MaskFormerImpl(**params).eval()
        res = model.load_state_dict(ckpt_data["state_dict"], strict=False) # inference only: we remove criterion
        assert res.unexpected_keys == ["criterion.empty_weight"] or res.unexpected_keys == [], res
        model.to("cuda" if tr.cuda.is_available() else "cpu")
        assert len(cfg.DATASETS.TEST) == 1, cfg.DATASETS.TEST
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        logger.debug(f"Loade weights from '{weights_path}'")
        return model, cfg, metadata

    def __del__(self):
        del self.model

def main():
    """main fn. Usage: python mask2former.py 49189528_1/47429163_0/49189528_0 demo1.jpg output1.jpg"""
    from datetime import datetime # pylint: disable=all
    from media_processing_lib.image import image_read, image_write # pylint: disable=all
    assert len(sys.argv) == 4
    img = image_read(sys.argv[2])

    m2f = Mask2Former(sys.argv[1], semantic_argmax_only=False, name="m2f", dependencies=[])
    m2f.model.to("cuda" if tr.cuda.is_available() else "cpu")
    for _ in range(1):
        now = datetime.now()
        pred = m2f.make(img[None])
        print(f"Pred took: {datetime.now() - now}")
        semantic_result = m2f.make_images(img[None], pred)[0]
        image_write(semantic_result, sys.argv[3])

    # Sanity checks
    if m2f.model_id == "47429163_0" and Path(sys.argv[2]).name == "demo1.jpg":
        assert np.allclose(mean := semantic_result.mean(), 129.41173), (mean, semantic_result.std())
        assert np.allclose(std := semantic_result.std(), 53.33731), std
    elif m2f.model_id == "49189528_1" and Path(sys.argv[2]).name == "demo1.jpg":
        assert np.allclose(mean := semantic_result.mean(), 125.23281), (mean, semantic_result.std())
        assert np.allclose(std := semantic_result.std(), 48.89948), std
    elif m2f.model_id == "49189528_0" and Path(sys.argv[2]).name == "demo1.jpg":
        assert np.allclose(mean := semantic_result.mean(), 118.47982), (mean, semantic_result.std())
        assert np.allclose(std := semantic_result.std(), 52.08105), std

if __name__ == "__main__":
    main()
