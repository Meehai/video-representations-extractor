import os
import gdown
import numpy as np
import torch as tr
import cv2
from pathlib import Path
from media_processing_lib.image import image_resize
from typing import List
from overrides import overrides

from .model_dexined import DexiNed as Model
from ...representation import Representation, RepresentationOutput
from ....logger import logger

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")


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
    img = image_resize(image, width=512, height=512)
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
    def __init__(self, **kwargs):
        self.model = None
        super().__init__(**kwargs)

    @overrides
    def setup(self):
        # our backup weights
        weightsFile = Path(f"{os.environ['VRE_WEIGHTS_DIR']}/dexined.pth").absolute()
        urlWeights = "https://drive.google.com/u/0/uc?id=1oT1iKdRRKJpQO-DTYWUnZSK51QnJ-mnP"

        if not weightsFile.exists():
            logger.debug(f"Downloading weights for dexined from {urlWeights}")
            gdown.download(urlWeights, str(weightsFile))

        model = Model().to(device)
        model.load_state_dict(tr.load(weightsFile, map_location=device))
        logger.debug2(f"Loaded weights from '{weightsFile}'")
        self.model = model

    @overrides
    def make(self, t: int) -> RepresentationOutput:
        A = preprocessImage(self.video[t])
        trA = tr.from_numpy(A.copy()).float()[None].to(device)
        with tr.no_grad():
            trB = self.model.forward(trA)
        C = postprocessOutput(trB)
        return C

    @overrides
    def make_image(self, x: RepresentationOutput) -> np.ndarray:
        x = np.repeat(np.expand_dims(x["data"], axis=-1), 3, axis=-1)
        return np.uint8(x * 255)
