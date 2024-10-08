"""FastSAM representation."""
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
import numpy as np
import torch as tr
from overrides import overrides
from torch.nn import functional as F

from vre.representation import Representation, RepresentationOutput
from vre.utils import image_resize_batch, fetch_weights, image_read, image_write
from vre.logger import vre_logger as logger
from vre.representations.soft_segmentation.fastsam.fastsam_impl import FastSAM as Model, FastSAMPredictor, FastSAMPrompt
from vre.representations.soft_segmentation.fastsam.fastsam_impl.results import Results
from vre.representations.soft_segmentation.fastsam.fastsam_impl.utils import bbox_iou
from vre.representations.soft_segmentation.fastsam.fastsam_impl.ops import \
    scale_boxes, non_max_suppression, process_mask_native

class FastSam(Representation):
    """FastSAM representation."""
    def __init__(self, variant: str, iou: float, conf: float, **kwargs):
        assert variant in ("fastsam-s", "fastsam-x"), variant
        self.variant = variant
        self.conf = conf
        self.iou = iou

        super().__init__(**kwargs)
        weights_path = self._get_weights_path(variant)
        model = Model(weights_path)
        model.model.eval()

        retina_masks = False
        imgsz = 1024 # TODO: see if this can be changed
        _overrides = {"task": "segment", "imgsz": imgsz, "single_cls": False,
                      "model": weights_path, "conf": conf, "device": "cpu", "retina_masks": retina_masks, "iou": iou,
                      "mode": "predict", "save": False}
        self.predictor = FastSAMPredictor(overrides=_overrides)
        self.predictor.setup_model(model=model.model, verbose=False)

    def _get_weights_path(self, variant: str) -> str:
        weights_file = {
            "fastsam-s": fetch_weights(__file__) / "FastSAM-s.pt",
            "fastsam-x": fetch_weights(__file__) / "FastSAM-x.pt"
        }[variant]
        return f"{weights_file}"

    @overrides
    def vre_setup(self):
        self.predictor.model = self.predictor.model.to(self.device)

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, RepresentationOutput] | None = None) -> RepresentationOutput:
        tr_x = self._preproces(frames)
        tr_y = self.predictor.model.forward(tr_x)
        mb, _, i_h, i_w = tr_x.shape[0:4]
        boxes = self._postprocess(preds=tr_y, inference_height=i_h, inference_width=i_w, conf=self.conf, iou=self.iou)
        extra = [{"boxes": boxes[i].to("cpu").numpy(), "inference_size": (i_h, i_w)} for i in range(mb)]
        assert len(tr_y[1]) == 3, len(tr_y[1])
        # Note: this is called 'proto' in the original implementation. Only this part of predictions is used for plots.
        res = tr_y[1][-1].to("cpu").numpy()
        return RepresentationOutput(output=res, extra=extra)

    @overrides(check_signature=False)
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        y_fastsam, extra = repr_data.output, repr_data.extra
        assert len(frames) == len(extra) == len(y_fastsam), (len(frames), len(extra), len(y_fastsam))
        assert all(e["inference_size"] == extra[0]["inference_size"] for e in extra), extra
        frames_rsz = image_resize_batch(frames, *self.size(repr_data))
        frame_h, frame_w = frames_rsz.shape[1:3]

        tr_y = tr.from_numpy(y_fastsam).to(self.device)
        boxes = [tr.from_numpy(e["boxes"]).to(self.device) for e in extra]
        scaled_boxes = [self._scale_box(box, *extra[0]["inference_size"], frame_h, frame_w) for box in boxes]

        res: list[np.ndarray] = []
        for i, (scaled_box, frame) in enumerate(zip(scaled_boxes, frames_rsz)):
            if len(scaled_box) == 0:  # save empty boxes
                res.append(frame)
                continue
            masks = process_mask_native(tr_y[i], scaled_box[:, 6:], scaled_box[:, :4], (frame_h, frame_w))
            res_i = Results(orig_img=frame, path=None, names={0: "object"}, boxes=scaled_box[:, 0:6], masks=masks)
            prompt_process = FastSAMPrompt(frame, [res_i], device=self.device)
            ann = prompt_process.results[0].masks.data
            prompt_res = prompt_process.plot_to_result(annotations=ann, better_quality=False, withContours=False)
            res.append(prompt_res)
        res_arr = np.array(res)
        return res_arr

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.extra[0]["inference_size"]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        y_fastsam, extra = repr_data.output, repr_data.extra
        old_size = extra[0]["inference_size"]
        new_extra = [{"boxes": self._scale_box(tr.from_numpy(e["boxes"]), *old_size, *new_size).numpy(),
                      "inference_size": new_size} for e in extra]
        new_y_fastsam = F.interpolate(tr.from_numpy(y_fastsam), (new_size[0] // 4, new_size[1] // 4)).numpy()
        return RepresentationOutput(output=new_y_fastsam, extra=new_extra)

    def _scale_box(self, box: tr.Tensor, inference_height: int, inference_width: int, original_height: int,
                   original_width: int) -> tr.Tensor:
        scaled_box = box.clone()
        if len(scaled_box) == 0:
            return scaled_box
        scaled_box[:, 0:4] = scale_boxes((inference_height, inference_width), scaled_box[:, 0:4],
                                         (original_height, original_width))
        return scaled_box

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
        desired_max = self.predictor.args.imgsz
        tr_x = tr.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        w, h = x.shape[1:3]
        if w > h:
            new_w, new_h = h * desired_max // w, desired_max
        else:
            new_w, new_h = desired_max, w * desired_max // h
        def _offset32(x: int) -> int:
            return ((32 - x % 32) if x < 32 or (x % 32 <= 16) else -(x % 32)) % 32 # get closest modulo 32 offset of x
        new_h, new_w = new_h + _offset32(new_h), new_w + _offset32(new_w) # needed because it throws otherwise
        tr_x = F.interpolate(tr_x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return tr_x.to(next(self.predictor.model.parameters()).device)

    def vre_free(self):
        if str(self.device).startswith("cuda"):
            self.predictor.model.to("cpu")
            tr.cuda.empty_cache()

def get_args() -> Namespace:
    """cli args"""
    parser = ArgumentParser()
    parser = ArgumentParser()
    parser.add_argument("model_id", choices=["fastsam-x", "fastsam-s"])
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args()

def main(args: Namespace):
    """main fn. Usage: python fastsam.py fastsam-x/fastsam-s demo1.jpg output1.jpg"""
    img = image_read(args.input_image)

    fastsam = FastSam(args.model_id, iou=0.9, conf=0.4, name="fastsam", dependencies=[])
    fastsam.predictor.model.to("cuda" if tr.cuda.is_available() else "cpu")
    now = datetime.now()
    pred = fastsam.make(img[None])
    logger.info(f"Pred took: {datetime.now() - now}")
    semantic_result = fastsam.make_images(img[None], pred)[0]
    image_write(semantic_result, args.output_path)
    logger.info(f"Written result to '{args.output_path}'")

    output_path_rsz = args.output_path.parent / f"{args.output_path.stem}_rsz.{args.output_path.suffix}"
    pred_rsz = fastsam.resize(pred, img.shape[0:2])
    semantic_result_rsz = fastsam.make_images(img[None], pred_rsz)[0]
    image_write(semantic_result_rsz, output_path_rsz)
    logger.info(f"Written resized result to '{output_path_rsz}'")

    rtol = 1e-2 if tr.cuda.is_available() else 1e-5
    if args.input_image.name == "demo1.jpg" and args.model_id == "fastsam-s":
        assert np.allclose(a := pred.output.mean(), 0.8813972, rtol=rtol), a
        assert np.allclose(b := pred.extra[0]["boxes"].mean(), 46.200043, rtol=rtol), b
        assert np.allclose(a := pred_rsz.output.mean(), 0.88434076, rtol=rtol), a
        assert np.allclose(b := pred_rsz.extra[0]["boxes"].mean(), 28.541258, rtol=rtol), b
    elif args.input_image.name == "demo1.jpg" and args.model_id == "fastsam-x":
        assert np.allclose(a := pred.output.mean(), 0.80122584, rtol=rtol), a
        assert np.allclose(b := pred.extra[0]["boxes"].mean(), 49.640644, rtol=rtol), b
        assert np.allclose(a := pred_rsz.output.mean(), 0.80056435, rtol=rtol), a
        assert np.allclose(b := pred_rsz.extra[0]["boxes"].mean(), 30.688671, rtol=rtol), b

if __name__ == "__main__":
    main(get_args())
