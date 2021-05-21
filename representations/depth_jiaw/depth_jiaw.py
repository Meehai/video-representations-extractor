import numpy as np
import torch
from skimage.transform import resize as imresize
from nwmodule.utilities import device

from .DispResNet import DispResNet
from ..representation import Representation

def preprocessImage(img, size=None, multiples=(64, 64), device=torch.device('cuda')):
    if size is None:
        size = img.shape[:2]
    img = img.astype(np.float32)
    img = imresize(img, closest_fit(size, multiples))
    img = np.transpose(img, (2, 0, 1))
    img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.45) / 0.225).to(device)
    return img


def postprocessImage(y, size=None):
    if size is None:
        size = y.shape[2:4]
    y = y.cpu().numpy()[0, 0]
    dph = 1 / y
    dph = imresize(dph, size)
    return dph


def closest_fit(size, multiples):
    return [round(size[i] / multiples[i]) for i in range(len(multiples))]

class DepthJiaw(Representation):
    # def __init__(self, weightsFile:str, resNetLayers:int):
    def __init__(self):
        model = DispResNet().to(device)
        # model = DispResNet(resNetLayers, False).to(device)
        # weights = torch.load(weightsFile, map_location=device)
        # model.load_state_dict(weights['state_dict'])
        model.eval()
        self.model = model
        self.multiples = (64, 64)   # is it 32 tho?

    def make(self, x):
        x_ = preprocessImage(x, multiples=self.multiples, device=device)
        breakpoint()
        with torch.no_grad():
            y = self.model(x_)
        y = postprocessImage(y, x.shape[:2])
        return y

    def makeImage(self, x):
        # TODO wat dis
        return x

    def setup(self):
        pass