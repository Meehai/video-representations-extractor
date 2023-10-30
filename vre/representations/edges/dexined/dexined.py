import os
import gdown
import numpy as np
import torch as tr
import cv2
from pathlib import Path
from typing import List
from overrides import overrides

from .model_dexined import DexiNed as Model
from ....representation import Representation
from ....logger import logger

def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def preprocessImage(image: np.ndarray) -> np.ndarray:
    logger.debug2(f"Original shape: {image.shape}")
    img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = np.array(img, dtype=np.float32)
    mean_pixel_values = [103.939, 116.779, 123.68]
    img -= mean_pixel_values
    img = img.transpose((2, 0, 1))
    return img


def postprocessOutput(y: List[tr.Tensor]) -> np.ndarray:
    preds = []
    for i in range(len(y)):
        tmp_img = tr.sigmoid(y[i]).cpu().detach().numpy().squeeze()
        tmp_img = np.uint8(image_normalization(tmp_img, img_min=0, img_max=255))
        tmp_img = cv2.bitwise_not(tmp_img)
        preds.append(tmp_img)
    average = np.array(preds, dtype=np.float32) / 255
    average = np.mean(average, axis=0)
    return average


class DexiNed(Representation):
    def __init__(self, device: str, **kwargs):
        self.model = None
        self.device = device
        assert tr.cuda.is_available() or self.device == "cpu", "CUDA not available"
        super().__init__(**kwargs)
        self._setup()

    def _setup(self):
        # our backup weights
        weights_file = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/dexined.pth").absolute()
        url_weights = "https://drive.google.com/u/0/uc?id=1oT1iKdRRKJpQO-DTYWUnZSK51QnJ-mnP"

        if not weights_file.exists():
            logger.debug(f"Downloading weights for dexined from {url_weights}")
            gdown.download(url_weights, str(weights_file))

        model = Model()
        model.load_state_dict(tr.load(weights_file, map_location="cpu"))
        logger.debug2(f"Loaded weights from '{weights_file}'")
        self.model = model.to(self.device)

    @overrides
    def make(self, t: int) -> np.ndarray:
        A = preprocessImage(self.video[t])
        trA = tr.from_numpy(A.copy()).float()[None].to(self.device)
        with tr.no_grad():
            trB = self.model.forward(trA)
        C = postprocessOutput(trB)
        return C

    @overrides
    def make_image(self, x: np.ndarray) -> np.ndarray:
        x = np.repeat(np.expand_dims(x["data"], axis=-1), 3, axis=-1)
        return np.uint8(x * 255)
