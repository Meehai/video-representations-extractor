"""Mask2Former representation"""
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from overrides import overrides
import torch as tr
import numpy as np

from vre.logger import vre_logger as logger
from vre.utils import (image_resize_batch, fetch_weights, image_read, image_write,
                       vre_load_weights, colorize_semantic_segmentation, VREVideo, FakeVideo, MemoryData)
from vre.representations import (
    Representation, ReprOut, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation)
from vre.representations.semantic_segmentation.mask2former.mask2former_impl import MaskFormer, CfgNode, get_output_shape

class Mask2Former(Representation, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation):
    """Mask2Former representation implementation. Note: only semantic segmentation (not panoptic/instance) enabled."""
    def __init__(self, model_id: str, semantic_argmax_only: bool, **kwargs):
        Representation.__init__(self, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        assert isinstance(model_id, str) and model_id in {"47429163_0", "49189528_1", "49189528_0"}, model_id
        self._m2f_resources = Path(__file__).parent / "mask2former_impl/resources"
        self.classes, self.color_map, self.thing_dataset_id_to_contiguous_id = self._get_metadata(model_id)
        self.semantic_argmax_only = semantic_argmax_only
        self.model_id = model_id
        self.model: MaskFormer | None = None
        self.cfg: CfgNode | None = None
        self.output_dtype = "uint8" if semantic_argmax_only else "float16"

    @tr.no_grad()
    @overrides
    def compute(self, video: VREVideo, ixs: list[int]):
        assert self.data is None, f"[{self}] data must not be computed before calling this"
        height, width = video.frame_shape[0:2]
        _os = get_output_shape(height, width, self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        imgs = image_resize_batch(video[ixs], _os[0], _os[1], "bilinear", "PIL").transpose(0, 3, 1, 2).astype("float32")
        inputs = [{"image": tr.from_numpy(img), "height": height, "width": width} for img in imgs]
        predictions: list[tr.Tensor] = [x["sem_seg"] for x in self.model(inputs)]
        res = []
        for pred in predictions:
            _pred = pred.argmax(dim=0) if self.semantic_argmax_only else pred.permute(1, 2, 0)
            res.append(_pred.to("cpu").numpy())
        self.data = ReprOut(frames=video[ixs], output=MemoryData(res), key=ixs)

    @overrides
    def make_images(self) -> np.ndarray:
        assert self.data is not None, f"[{self}] data must be first computed using compute()"
        assert self.data.frames is not None and self.data.output is not None, self.data
        res = []
        frames_rsz = image_resize_batch(self.data.frames, *self.data.output.shape[1:3])
        for img, pred in zip(frames_rsz, self.data.output):
            _pred = pred if self.semantic_argmax_only else pred.argmax(-1)
            res.append(colorize_semantic_segmentation(_pred, self.classes, self.color_map, img))
        res = np.stack(res)
        return res

    @overrides
    def vre_setup(self, load_weights = True):
        assert self.setup_called is False
        weights_path = fetch_weights(__file__) / f"{self.model_id}.ckpt"
        assert isinstance(weights_path, Path), type(weights_path)
        ckpt_data = vre_load_weights(weights_path)
        self.cfg = CfgNode(json.load(open(f"{self._m2f_resources}/{self.model_id}_cfg.json", "r")))
        params = MaskFormer.from_config(self.cfg)
        params["metadata"] = SimpleNamespace(thing_dataset_id_to_contiguous_id=self.thing_dataset_id_to_contiguous_id)
        self.model = MaskFormer(**{**params, "semantic_on": True, "panoptic_on": False, "instance_on": False})
        if load_weights:
            res = self.model.load_state_dict(ckpt_data["state_dict"], strict=False) # inference only: remove criterion
            assert res.unexpected_keys in (["criterion.empty_weight"], []), res
        self.model = self.model.eval().to(self.device)
        self.setup_called = True

    @overrides
    def vre_free(self):
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
        self.model = None
        self.setup_called = False

    def _get_metadata(self, model_id: str) -> tuple[list[str], list[tuple[int, int, int]], dict[str, int]]:
        metadata = None
        if model_id == "49189528_1": # r50
            metadata = json.load(open(f"{self._m2f_resources}/mapillary_metadata.json", "r"))
        if model_id == "47429163_0": # swin
            metadata = json.load(open(f"{self._m2f_resources}/coco_metadata.json", "r"))
        if model_id == "49189528_0": # swin
            metadata = json.load(open(f"{self._m2f_resources}/mapillary_metadata2.json", "r"))
        return metadata["stuff_classes"], metadata["stuff_colors"], metadata.get("thing_dataset_id_to_contiguous_id")

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
    m2f.compute(FakeVideo(img[None], 1), [0])
    logger.info(f"Pred took: {datetime.now() - now}")
    semantic_result: np.ndarray = m2f.make_images()[0]
    image_write(semantic_result, args.output_path)
    logger.info(f"Written prediction to '{args.output_path}'")
    return semantic_result # for integration tests

if __name__ == "__main__":
    main(get_args())
