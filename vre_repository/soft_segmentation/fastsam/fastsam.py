"""FastSAM representation."""
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from overrides import overrides
import numpy as np
import torch as tr
from torch import nn
from torch.nn import functional as F

from vre_video import VREVideo
from vre.representations import (Representation, ReprOut, LearnedRepresentationMixin,
                                 NpIORepresentation, NormedRepresentationMixin)
from vre.utils import image_resize_batch, image_read, image_write, MemoryData
from vre.logger import vre_logger as logger
from vre_repository.weights_repository import fetch_weights

try:
    from .fastsam_impl import (
        FastSAM as Model, FastSAMPredictor, FastSAMPrompt, Results, bbox_iou, non_max_suppression, process_mask_native)
except ImportError:
    from vre_repository.soft_segmentation.fastsam.fastsam_impl import (
        FastSAM as Model, FastSAMPredictor, FastSAMPrompt, Results, bbox_iou, non_max_suppression, process_mask_native)

class FastSam(Representation, LearnedRepresentationMixin, NpIORepresentation, NormedRepresentationMixin):
    """FastSAM representation."""
    def __init__(self, variant: str, iou: float, conf: float, **kwargs):
        Representation.__init__(self, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        NormedRepresentationMixin.__init__(self)
        assert variant in ("fastsam-s", "fastsam-x", "testing"), variant
        self.variant = variant
        self.conf = conf
        self.iou = iou
        self.model: nn.Module | None = None
        self._imgsz: int | None = None

    @overrides
    def compute(self, video: VREVideo, ixs: list[int], dep_data: list[ReprOut] | None = None) -> ReprOut:
        tr_x = self._preproces(frames := video[ixs])
        tr_y: tr.FloatTensor = self.model.forward(tr_x)
        mb, _, i_h, i_w = tr_x.shape[0:4]
        boxes = self._postprocess(preds=tr_y, inference_height=i_h, inference_width=i_w, conf=self.conf, iou=self.iou)
        # Note: image_size' see the commnet in resize()
        extra = [{"boxes": boxes[i].to("cpu").numpy(), "inference_size": (i_h, i_w), "image_size": frames.shape[1:3]}
                 for i in range(mb)]
        assert len(tr_y[1]) == 3, len(tr_y[1])
        # Note: this is called 'proto' in the original implementation. Only this part of predictions is used for plots.
        return ReprOut(frames=frames, output=MemoryData(tr_y[1][-1].to("cpu").numpy()), extra=extra, key=ixs)

    @overrides(check_signature=False)
    def make_images(self, data: ReprOut) -> np.ndarray:
        y_fastsam, extra = data.output, data.extra # len(y_fastsam) == len(extra) == len(ixs)
        assert all((inf_size := e["inference_size"]) == extra[0]["inference_size"] for e in extra), extra
        assert all((img_size := e["image_size"]) == extra[0]["image_size"] for e in extra), extra
        tr_y = tr.from_numpy(y_fastsam).to(self.device)
        boxes = [tr.from_numpy(e["boxes"]).to(self.device) for e in extra]
        # scaled_boxes = [scale_box(box, *inf_size, *img_size) for box in boxes] # TODO: needed for faster proc?

        res: list[np.ndarray] = []
        for i, (box, frame) in enumerate(zip(boxes, data.frames)):
            if len(box) == 0:  # save empty boxes
                res.append(frame)
                continue
            masks = process_mask_native(tr_y[i], box[:, 6:], box[:, :4], inf_size)
            res_i = Results(orig_img=frame, path=None, names={0: "object"}, boxes=box[:, 0:6], masks=masks)
            prompt_process = FastSAMPrompt(frame, [res_i], device=self.device)
            ann = prompt_process.results[0].masks.data
            prompt_res = prompt_process.plot_to_result(annotations=ann, withContours=True)
            res.append(prompt_res)
        res_arr = image_resize_batch(res, *img_size, interpolation="bilinear")
        return res_arr

    @overrides
    def size(self, repr_out: ReprOut) -> tuple[int, ...]:
        return (len(repr_out.output), *repr_out.extra[0]["image_size"], 3) # Note: not the embeddings size!

    @overrides
    def resize(self, data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        assert data is not None
        # Note: check the old implementation. We used to rescale the boxes, now we just keep them as-is and
        # only resize the final output. It's debatable if we want to resize the boxes or not to maintain quality.
        new_extra = [{**e, "image_size": new_size} for e in data.extra]
        output_images = None
        if data.output_images is not None:
            output_images = image_resize_batch(data.output_images, *new_size, interpolation="nearest")
        return ReprOut(frames=data.frames, output=data.output, extra=new_extra,
                       key=data.key, output_images=output_images)

    @staticmethod
    @overrides
    def weights_repository_links(**kwargs) -> list[str]:
        match kwargs["variant"]:
            case "fastsam-x": return ["soft_segmentation/fastsam/FastSAM-x.pt"]
            case "fastsam-s": return ["soft_segmentation/fastsam/FastSAM-s.pt"]
            case "testing": return []
            case _: raise NotImplementedError(kwargs)

    @overrides
    def vre_setup(self, load_weights: bool = True):
        assert self.setup_called is False
        assert load_weights is False or (load_weights is True and self.variant != "testing"), "no weights for testing"
        weights_file = "testing"
        if self.variant != "testing":
            weights_file = fetch_weights(FastSam.weights_repository_links(variant=self.variant))[0]
        self._imgsz = 1024 if self.variant != "testing" else 64
        _overrides = {"task": "segment", "imgsz": self._imgsz, "single_cls": False, "model": str(weights_file),
                      "conf": self.conf, "device": "cpu", "retina_masks": False, "iou": self.iou,
                      "mode": "predict", "save": False}
        predictor = FastSAMPredictor(overrides=_overrides)
        _model = Model(str(weights_file))
        predictor.setup_model(model=_model.model, verbose=False)
        self.model = predictor.model.eval().to(self.device)
        self.setup_called = True

    @overrides
    def vre_free(self):
        assert self.setup_called is True and self.model is not None, (self.setup_called, self.model is not None)
        if str(self.device).startswith("cuda"):
            self.model.to("cpu")
            tr.cuda.empty_cache()
        self.model = None
        self.setup_called = False

    @property
    @overrides
    def n_channels(self) -> int:
        raise ValueError("I hate inheritance. Makes no sense for this representation")

    def _postprocess(self, preds: tr.Tensor, inference_height: int, inference_width: int,
                     conf: float, iou: float) -> list[tr.Tensor]:
        p = non_max_suppression(preds[0], conf, iou, agnostic=False, max_det=300, nc=1, classes=None)

        for _p in p:
            if len(_p) == 0:
                continue
            full_box = tr.zeros_like(_p[0])
            full_box[2], full_box[3], full_box[4], full_box[6:] = inference_width, inference_height, 1.0, 1.0
            full_box = full_box.view(1, -1)
            critical_iou_index = bbox_iou(full_box[0][:4], _p[:, :4], iou_thres=0.9,
                                          image_shape=(inference_height, inference_width))
            if critical_iou_index.numel() != 0:
                full_box[0][4] = _p[critical_iou_index][:, 4]
                full_box[0][6:] = _p[critical_iou_index][:, 6:]
                _p[critical_iou_index] = full_box
        return p

    def _preproces(self, x: np.ndarray) -> tr.Tensor:
        x = x.astype(np.float32) / 255
        desired_max = self._imgsz
        tr_x = tr.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        w, h = x.shape[1:3]
        if w > h:
            new_w, new_h = h * desired_max // w, desired_max
        else:
            new_w, new_h = desired_max, w * desired_max // h
        def _offset32(x: int) -> int:
            return ((32 - x % 32) if x < 32 or (x % 32 <= 16) else -(x % 32)) % 32 # get closest modulo 32 offset of x
        new_h, new_w = new_h + _offset32(new_h), new_w + _offset32(new_w) # needed because it throws otherwise
        tr_x: tr.FloatTensor = F.interpolate(tr_x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return tr_x.to(self.device)

def get_args() -> Namespace:
    """cli args"""
    parser = ArgumentParser()
    parser = ArgumentParser()
    parser.add_argument("model_id", choices=["fastsam-x", "fastsam-s", "testing"])
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args()

def main(args: Namespace):
    """main fn. Usage: python fastsam.py fastsam-x/fastsam-s demo1.jpg output1.jpg"""
    img = image_read(args.input_image)

    fastsam = FastSam(args.model_id, iou=0.9, conf=0.4, name="fastsam", dependencies=[])
    fastsam.device = "cuda" if tr.cuda.is_available() else "cpu"
    fastsam.vre_setup(load_weights=args.model_id != "testing")
    now = datetime.now()
    pred = fastsam.compute(VREVideo(img[None], fps=1), [0])
    logger.info(f"Pred took: {datetime.now() - now}")
    semantic_result = fastsam.make_images(pred)[0]
    image_write(semantic_result, args.output_path)
    logger.info(f"Written result to '{args.output_path}'")

    output_path_rsz = args.output_path.parent / f"{args.output_path.stem}_rsz{args.output_path.suffix}"
    pred_rsz = fastsam.resize(pred, (300, 1024))
    semantic_result_rsz = fastsam.make_images(pred_rsz)[0]
    image_write(semantic_result_rsz, output_path_rsz)
    logger.info(f"Written resized result to '{output_path_rsz}'")

    rtol = 1e-2 if tr.cuda.is_available() else 1e-5
    if args.input_image.name == "demo1.jpg" and args.model_id == "fastsam-s":
        assert np.allclose(a := pred.output.mean(), 0.8813972, rtol=rtol), a
        assert np.allclose(b := pred.extra[0]["boxes"].mean(), 46.200043, rtol=rtol), b
        # Note: now only the tuple size and make_images is changed. To be decided if it's worth to downscale preds.
        assert np.allclose(a := pred_rsz.output.mean(), 0.8813972, rtol=rtol), a
        assert np.allclose(b := pred_rsz.extra[0]["boxes"].mean(), 46.200043, rtol=rtol), b
        logger.info("Mean and std test passed!")
    elif args.input_image.name == "demo1.jpg" and args.model_id == "fastsam-x":
        assert np.allclose(a := pred.output.mean(), 0.80122584, rtol=rtol), a
        assert np.allclose(b := pred.extra[0]["boxes"].mean(), 49.640644, rtol=rtol), b
        assert np.allclose(a := pred_rsz.output.mean(), 0.80122584, rtol=rtol), a
        assert np.allclose(b := pred_rsz.extra[0]["boxes"].mean(), 49.640644, rtol=rtol), b
        logger.info("Mean and std test passed!")

if __name__ == "__main__":
    main(get_args())
