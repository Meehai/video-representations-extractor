# pylint: disable=all
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
FastSAM model interface.

Usage - Predict:
    from ultralytics import FastSAM

    model = FastSAM('last.pt')
    results = model.predict('ultralytics/assets/bus.jpg')
"""

from torch import nn
import torch
from pathlib import Path
import sys
sys.path.insert(0, Path(__file__).parents[1].__str__()) # needed to load the weights because ultralytics must be in path

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.modules import Segment
from .predict import FastSAMPredictor
from .utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class FakeSAM(SegmentationModel):
    def __init__(self):
        super().__init__()
        self.head = Segment(nc=1, nm=32, npr=128, ch=(128, 256, 512), reg_max=26)

    @torch.no_grad
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mb, device = len(x), next(self.head.parameters()).device
        # [tensor[1, 320, 72, 128] n=2949120 (11Mb) x∈[-0.278, 6.678] μ=0.216 σ=0.407 cuda:0,
        #  tensor[1, 640, 36, 64] n=1474560 (5.6Mb) x∈[-0.278, 9.127] μ=0.034 σ=0.335 cuda:0,
        #  tensor[1, 640, 18, 32] n=368640 (1.4Mb) x∈[-0.278, 7.905] μ=0.173 σ=0.529 cuda:0]
        x = [
            torch.randn(mb, 128, 64, 128).to(device),
            torch.randn(mb, 256, 32, 64).to(device),
            torch.randn(mb, 512, 16, 32).to(device)
        ]
        y = self.head(x)
        return y

class FastSAM:

    def __init__(self, model: str | Path, task=None) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        # self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        model = str(model).strip()  # strip spaces

        # Load or create new YOLO model
        self._load(model, task)

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
        if weights == "testing":
            model = FakeSAM()
            args = DEFAULT_CFG_DICT
        else:
            # self.model, _ = attempt_load_one_weight(weights)
            ckpt = torch.load(weights)
            args = {**DEFAULT_CFG_DICT, **(ckpt.get('train_args', {}))}  # combine model and default args, preferring model args
            model = (ckpt.get('ema') or ckpt['model']).to("cpu").float()  # FP32 model

            # Model compatibility updates
            model.pt_path = weights  # attach *.pt file path to model
            if not hasattr(model, 'stride'):
                model.stride = torch.tensor([32.])

            model = model.fuse().eval() if hasattr(model, 'fuse') else model.eval()  # model in eval mode

            # Module compatibility updates
            for m in model.modules():
                t = type(m)
                if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU):
                    m.inplace = True  # torch 1.7.0 compatibility
                elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                    m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        model.task = "segment"
        self.model = model

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    @torch.no_grad()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        assert source is not None
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        self.predictor = FastSAMPredictor(overrides=overrides)
        self.predictor.setup_model(model=self.model, verbose=False)
        try:
            return self.predictor(source, stream=stream)
        except Exception:
            return None

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")
