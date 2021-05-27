import sys
import gdown
import numpy as np
import torch as tr
from shutil import copyfile
from nwdata.utils import fullPath
from nwmodule.utilities import device, trModuleWrapper
from media_processing_lib.image import imgResize, tryWriteImage
from scipy.special import expit as sigmoid

from .model_dexined import DexiNed as Model
from ..representation import Representation

def preprocessImage(img):
    img, coordinates = imgResize(img, height=352, width=352, mode="black_bars", \
        resizeLib="lycon", returnCoordinates=True)
    img = np.float32(img) / 255
    img = img.transpose(2, 0, 1)[None]
    return img, coordinates

def postprocessImage(img, coordinates):
    img = np.array(img)
    img = sigmoid(img)
    img[img < 0.0] = 0.0
    img = 1 - img
    img = img.mean(axis=0)[0][0]
    x0, y0, x1, y1 = coordinates
    img = img[y0 : y1, x0 : x1]
    return img

class DexiNed(Representation):
    def __init__(self):
        self.weightsFile = str(fullPath(__file__).parents[0] / "weights.pth")
        self.setup()
        model = Model().to(device)
        model.load_state_dict(tr.load(self.weightsFile, map_location=device))
        self.model = trModuleWrapper(model)

    def setup(self):
        urlWeights = "https://drive.google.com/u/0/uc?id=1MRUlg_mRwDiBiQLKFVuEfuvkzs65JFVe"

        weightsPath = fullPath(self.weightsFile)
        if not weightsPath.exists():
            print("[DexiNed::setup] Downloading weights for dexined from %s" % urlWeights)
            gdown.download(urlWeights, self.weightsFile)

    def make(self, video, t):
        A, coordinates = preprocessImage(video[t])
        with tr.no_grad():
            B = self.model.npForward(A)
        C = postprocessImage(B, coordinates)
        return C

    def makeImage(self, x):
        x = np.repeat(np.expand_dims(x, axis=-1), 3, axis=-1)
        return np.uint8(x * 255)
