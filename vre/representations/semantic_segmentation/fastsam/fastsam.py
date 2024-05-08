"""FastSAM representation."""
from overrides import overrides
import numpy as np
import torch as tr
from torch.nn import functional as F

from ultralytics.yolo.engine.results import Results # TODO: get rid of these
from ultralytics.yolo.utils import ops # TODO: get rid of these

try:
    from .fastsam_impl import FastSAM as Model, FastSAMPredictor, FastSAMPrompt
    from .fastsam_impl.utils import bbox_iou
    from ....representation import Representation, RepresentationOutput
    from ....utils import gdown_mkdir, VREVideo, image_resize_batch, get_weights_dir
except ImportError:
    from fastsam_impl import FastSAM as Model, FastSAMPredictor, FastSAMPrompt
    from fastsam_impl.utils import bbox_iou
    from vre.representation import Representation, RepresentationOutput
    from vre.utils import gdown_mkdir, VREVideo, image_resize_batch, get_weights_dir

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
        weights_path = get_weights_dir() / f"FastSAM-{'s' if variant == 'fastsam-s' else 'x'}.pt"
        links = {
            "fastsam-s": "https://drive.google.com/u/0/uc?id=1DlAy02fyGpRQHZThVEEpwKdlXKphSHbk",
            "fastsam-x": "https://drive.google.com/u/0/uc?id=1B2rGiBYCjk4B-jHMNzwgywpIdQjtKVRI",
        }
        if not weights_path.exists():
            gdown_mkdir(links[variant], weights_path)
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
        mb, _, i_h, i_w = tr_x.shape[0:4]
        boxes = self._postprocess(preds=tr_y, inference_height=i_h, inference_width=i_w, conf=self.conf, iou=self.iou)
        extra = [{"boxes": boxes[i].to("cpu").numpy(), "inference_size": (i_h, i_w)} for i in range(mb)]
        assert len(tr_y[1]) == 3, len(tr_y[1])
        # Note: this is called 'proto' in the original implementation. Only this part of predictions is used for plots.
        res = tr_y[1][-1].to("cpu").numpy()
        return res, extra

    @overrides(check_signature=False)
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        y_fastsam, extra = repr_data
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
            masks = ops.process_mask_native(tr_y[i], scaled_box[:, 6:], scaled_box[:, :4], (frame_h, frame_w))
            res_i = Results(orig_img=frame, path=None, names={0: "object"}, boxes=scaled_box[:, 0:6], masks=masks)
            prompt_process = FastSAMPrompt(frame, [res_i], device=self.device)
            ann = prompt_process.results[0].masks.data
            prompt_res = prompt_process.plot_to_result(annotations=ann, better_quality=False, withContours=False)
            res.append(prompt_res)
        res_arr = np.array(res)
        return res_arr

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data[1][0]["inference_size"]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        y_fastsam, extra = repr_data
        old_size = extra[0]["inference_size"]
        new_extra = [{"boxes": self._scale_box(tr.from_numpy(e["boxes"]), *old_size, *new_size).numpy(),
                      "inference_size": new_size} for e in extra]
        new_y_fastsam = F.interpolate(tr.from_numpy(y_fastsam), (new_size[0] // 4, new_size[1] // 4)).numpy()
        return new_y_fastsam, new_extra

    def _scale_box(self, box: tr.Tensor, inference_height: int, inference_width: int, original_height: int,
                   original_width: int) -> tr.Tensor:
        scaled_box = box.clone()
        if len(scaled_box) == 0:
            return scaled_box
        scaled_box[:, 0:4] = ops.scale_boxes((inference_height, inference_width), scaled_box[:, 0:4],
                                             (original_height, original_width))
        return scaled_box

    def _postprocess(self, preds: tr.Tensor, inference_height: int, inference_width: int,
                     conf: float, iou: float) -> list[tr.Tensor]:
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
        return tr_x

def main():
    """main fn. Usage: python fastsam.py fastsam-x/fastsam-s demo1.jpg output1.jpg"""
    import sys
    from datetime import datetime # pylint: disable=all
    from media_processing_lib.image import image_read, image_write # pylint: disable=all
    assert len(sys.argv) == 4
    img = image_read(sys.argv[2])

    fastsam = FastSam(sys.argv[1], iou=0.9, conf=0.4, name="fastsam", dependencies=[])
    fastsam.predictor.model.to("cuda" if tr.cuda.is_available() else "cpu")
    for _ in range(1):
        now = datetime.now()
        pred = fastsam.make(img[None])
        print(f"Pred took: {datetime.now() - now}")
        semantic_result = fastsam.make_images(img[None], pred)[0]
        image_write(semantic_result, sys.argv[3])

    pred_rsz = fastsam.resize(pred, img.shape[0:2])
    semantic_result_rsz = fastsam.make_images(img[None], pred_rsz)[0]
    image_write(semantic_result_rsz, f"{sys.argv[3][0:-4]}_rsz.{sys.argv[3][-3:]}")

if __name__ == "__main__":
    main()
