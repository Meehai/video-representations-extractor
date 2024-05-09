"""FlowRife representation"""
import numpy as np
import torch as tr
import torch.nn.functional as F
import flow_vis
from overrides import overrides

from ....utils import gdown_mkdir, image_resize_batch, VREVideo, get_weights_dir
from ....representation import Representation, RepresentationOutput
from .rife_impl.RIFE_HDv2 import Model

class FlowRife(Representation):
    """FlowRife representation"""
    def __init__(self, compute_backward_flow: bool, uhd: bool, **kwargs):
        tr.manual_seed(42)
        self.model: Model = Model().eval().to("cpu")
        self.uhd = uhd
        assert compute_backward_flow is False, "Not supported"
        self.no_backward_flow = True if compute_backward_flow is None else not compute_backward_flow
        self.device = "cpu"
        assert tr.cuda.is_available() or self.device == "cpu", "CUDA not available"
        super().__init__(**kwargs)

    # pylint: disable=arguments-differ
    @overrides(check_signature=False)
    def vre_setup(self, video: VREVideo, device: str):
        self.device = device
        weights_dir = get_weights_dir() / "rife"
        weights_dir.mkdir(exist_ok=True, parents=True)

        # original files
        # urlWeights = "https://drive.google.com/u/0/uc?id=1wsQIhHZ3Eg4_AfCXItFKqqyDMB4NS0Yd"
        # our backup / dragos' better/sharper version
        contextnet_url = "https://drive.google.com/u/0/uc?id=1x2_inKGBxjTYvdn58GyRnog0C7YdzE7-"
        flownet_url = "https://drive.google.com/u/0/uc?id=1aqR0ciMzKcD-N4bwkTK8go5FW4WAKoWc"
        unet_url = "https://drive.google.com/u/0/uc?id=1Fv27pNAbrmqQJolCFkD1Qm1RgKBRotME"

        contextnet_path = weights_dir / "contextnet.pkl"
        if not contextnet_path.exists():
            gdown_mkdir(contextnet_url, contextnet_path)

        flownet_path = weights_dir / "flownet.pkl"
        if not flownet_path.exists():
            gdown_mkdir(flownet_url, flownet_path)

        unet_path = weights_dir / "unet.pkl"
        if not unet_path.exists():
            gdown_mkdir(unet_url, unet_path)

        self.model.load_model(weights_dir)
        self.model = self.model.eval().to(self.device)

    @overrides
    def vre_dep_data(self, video: VREVideo, ix: slice) -> dict[str, RepresentationOutput]:
        right_frames = np.array(video[ix.start + 1: min(ix.stop + 1, len(video))])
        if ix.stop + 1 > len(video):
            right_frames = np.concatenate([right_frames, np.array([video[-1]])], axis=0)
        return {"right_frames": right_frames}

    # pylint: disable=arguments-differ
    @overrides(check_signature=False)
    def make(self, frames: np.ndarray, right_frames: np.ndarray) -> RepresentationOutput:
        x_s, x_t, padding = self._preprocess(frames, right_frames)
        with tr.no_grad():
            prediction = self.model.inference(x_s, x_t, self.uhd, self.no_backward_flow)
        flow = self._postprocess(prediction, padding)
        return flow

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: RepresentationOutput) -> np.ndarray:
        y = np.array([flow_vis.flow_to_color(_pred) for _pred in repr_data])
        return y

    @overrides
    def size(self, repr_data: RepresentationOutput) -> tuple[int, int]:
        return repr_data.shape[1:3]

    @overrides
    def resize(self, repr_data: RepresentationOutput, new_size: tuple[int, int]) -> RepresentationOutput:
        return image_resize_batch(repr_data, *new_size)

    def _preprocess(self, sources: np.ndarray, targets: np.ndarray) -> (tr.Tensor, tr.Tensor, tuple):
        # Convert, preprocess & pad
        sources = sources.transpose(0, 3, 1, 2)
        targets = targets.transpose(0, 3, 1, 2)
        tr_sources = tr.from_numpy(sources).to(self.device).float() / 255.0
        tr_targets = tr.from_numpy(targets).to(self.device).float() / 255.0
        h, w = tr_sources.shape[2:4]
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        tr_sources_padded = F.pad(tr_sources, padding) # pylint: disable=not-callable
        tr_target_padded = F.pad(tr_targets, padding) # pylint: disable=not-callable
        return tr_sources_padded, tr_target_padded, padding

    def _postprocess(self, prediction: tr.Tensor, padding: tuple) -> np.ndarray:
        flow = prediction.cpu().numpy().transpose(0, 2, 3, 1) # (B, H, W, C)
        returned_shape = flow.shape[1:3]
        # Remove the padding to keep original shape
        half_ph, half_pw = padding[3] // 2, padding[1] // 2
        flow = flow[:, 0: returned_shape[0] - half_ph, 0: returned_shape[1] - half_pw]
        flow = flow / returned_shape # [-px : px] => [-1 : 1]
        return flow.astype(np.float16)
