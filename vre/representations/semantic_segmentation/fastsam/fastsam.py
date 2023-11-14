"""FastSAM representation."""
from pathlib import Path
import os
from overrides import overrides
import numpy as np
import torch as tr
from torch.nn import functional as F

from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from .fastsam_impl import FastSAM as Model, FastSAMPredictor, FastSAMPrompt
from .fastsam_impl.utils import bbox_iou

from ....representation import Representation, RepresentationOutput
from ....utils import gdown_mkdir, VREVideo

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
        self.device = "cpu"

    def _get_weights_path(self, variant: str) -> str:
        weights_dir = Path(f"{os.environ['VRE_WEIGHTS_DIR']}").absolute()
        weights_path = Path(f"{weights_dir}/FastSAM-{'s' if variant == 'fastsam-s' else 'x'}.pt")
        if not weights_path.exists():
            gdown_mkdir("https://drive.google.com/u/0/uc?id=1DlAy02fyGpRQHZThVEEpwKdlXKphSHbk", weights_path)
        return f"{weights_path}"

    # pylint: disable=unused-argument, arguments-differ
    @overrides(check_signature=False)
    def vre_setup(self, video: VREVideo, device: str):
        self.predictor.model = self.predictor.model.to(device)
        self.device = device

    @overrides
    def make(self, frames: np.ndarray) -> RepresentationOutput:
        tr_x = self._preproces(frames)
        tr_y = self.predictor.model.model(tr_x)
        scaled_boxes = self._postprocess(preds=tr_y, inference_height=tr_x.shape[2], inference_width=tr_x.shape[3],
                                         conf=self.conf, iou=self.iou, original_height=frames.shape[1],
                                         original_width=frames.shape[2])
        extra = [{"scaled_boxes": scaled_box.to("cpu").numpy()} for scaled_box in scaled_boxes]
        assert len(tr_y[1]) == 3
        # Note: this is called 'proto' in the original implementation. Only this part of predictions is used for plots.
        res = tr_y[1][-1].to("cpu").numpy()
        return res, extra

    @overrides(check_signature=False)
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        repr_data, extra = repr_data
        assert extra is not None and len(frames) == len(extra)
        res: list[np.ndarray] = []
        tr_y = tr.from_numpy(repr_data).to(self.device)
        scaled_boxes = [tr.from_numpy(e["scaled_boxes"]).to(self.device) for e in extra]
        y_predictor = self._postprocess2(tr_y, scaled_boxes, original_height=frames.shape[1],
                                         original_width=frames.shape[2], orig_imgs=frames)

        for i in range(len(y_predictor)):
            if len(y_predictor[i].boxes) == 0:
                res.append(y_predictor[i].orig_img)
                continue
            prompt_process = FastSAMPrompt(y_predictor[i].orig_img, y_predictor[i: i + 1], device=self.device)
            ann = prompt_process.results[0].masks.data
            prompt_res = prompt_process.plot_to_result(annotations=ann, better_quality=False, withContours=False)
            res.append(prompt_res)
        res_arr = np.array(res)
        return res_arr

    def _postprocess(self, preds: tr.Tensor, inference_height: int, inference_width: int,
                     conf: float, iou: float, original_height: int, original_width: int) -> list[tr.Tensor]:
        p = ops.non_max_suppression(preds[0], conf, iou, agnostic=False, max_det=300, nc=1, classes=None)

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

        scaled_boxes = p.copy()
        for i in range(len(scaled_boxes)):
            if len(scaled_boxes[i]) == 0:
                continue
            scaled_boxes[i][:, 0:4] = ops.scale_boxes((inference_height, inference_width), scaled_boxes[i][:, 0:4],
                                                      (original_height, original_width))
        return scaled_boxes

    def _postprocess2(self, proto: tr.Tensor, scaled_boxes: list, original_height: int,
                      original_width: int, orig_imgs: np.ndarray) -> list[Results]:
        results: list[Results] = []
        names = {0: "object"}
        for i, scaled_box in enumerate(scaled_boxes):
            if len(scaled_box) == 0:  # save empty boxes
                masks = None
            else:
                masks = ops.process_mask_native(proto[i], scaled_box[:, 6:], scaled_box[:, :4],
                                                (original_height, original_width))
            results.append(Results(orig_img=orig_imgs[i], path=None, names=names,
                                   boxes=scaled_box[:, 0:6], masks=masks))
        return results

    def _preproces(self, x: np.ndarray) -> tr.Tensor:
        x = x.astype(np.float32) / 255
        desired_max = self.predictor.args.imgsz
        tr_x = tr.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        w, h = x.shape[1:3]
        if w > h:
            new_w, new_h = h * desired_max // w, desired_max
        else:
            new_w, new_h = desired_max, w * desired_max // h

        tr_x = F.interpolate(tr_x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return tr_x
