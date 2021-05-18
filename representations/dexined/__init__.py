import numpy as np
import torch as tr
from nwdata.utils import fullPath
from nwmodule.utilities import device
from media_processing_lib.image import imgResize, tryWriteImage
from scipy.special import expit as sigmoid

from ..representation import Representation

import sys
sys.path.append(str(fullPath(__file__).parents[0] / "DexiNed"))
from model_dexined import DexiNed as Model

def preprocessImage(img):
    img, coordinates = imgResize(img, height=512, width=512, mode="black_bars", imgLib="lycon", returnCoordinates=True)
    img = np.float32(img) / 255
    img = img.transpose(2, 0, 1)[None]
    return img, coordinates

def postprocessImage(img, coordinates):
    img = np.array(img)
    img[img < 0.0] = 0.0
    img = 1 - img
    img = img.mean(axis=0)[0][0]
    x0, y0, x1, y1 = coordinates
    img = img[y0 : y1, x0 : x1]
    return img

class DexiNed(Representation):
    def __init__(self, weightsFile:str):
        self.model = Model().to(device)
        self.model.load_state_dict(tr.load(weightsFile, map_location=device))
        self.model.eval()

    def make(self, frame):
        A, coordinates = preprocessImage(frame)
        with tr.no_grad():
            B = self.model.forward(tr.from_numpy(A).to(device))
            C = [tr.sigmoid(x) for x in B]
            D = [x.cpu().numpy() for x in C]
        E = postprocessImage(D, coordinates)
        F = imgResize(E, height=frame.shape[0], width=frame.shape[1], onlyUint8=False)
        return F

    def makeImage(self, x):
        return np.uint8(x * 255)
