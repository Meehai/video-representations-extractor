from .representation import Representation

def getRepresentation(type:str, method:str) -> Representation:
    objType = None

    if type == "default":
        if method == "rgb":
            from .rgb import RGB
            objType = RGB
        elif method == "hsv":
            from .hsv import HSV
            objType = HSV

    elif type == "soft-segmentation":
        if method == "python-halftone":
            from .soft_segmentation.python_halftone import Halftone
            objType = Halftone
        elif method == "kmeans":
            from .soft_segmentation.kmeans import KMeans
            objType = KMeans
        elif method == "generalized_boundaries":
            from .soft_segmentation.generalized_boundaries import GeneralizedBoundaries
            objType = GeneralizedBoundaries

    elif type == "edges":
        if method == "dexined":
            from .edges.dexined import DexiNed
            objType = DexiNed
        elif method == "canny":
            from .edges.canny import Canny
            objType = Canny

    elif type == "depth":
        if method == "dpt":
            from .depth.dpt import DepthDpt
            objType = DepthDpt
        elif method == "odo-flow":
            from .depth.odo_flow import DepthOdoFlow
            objType = DepthOdoFlow
        elif method == "depth-dispresnet":
            from .depth.dispresnet import DepthDispResNet
            objType = DepthDispResNet

    elif type == "optical-flow":
        if method == "rife":
            from .optical_flow.rife import FlowRife
            objType = FlowRife
        elif method == "raft":
            from .optical_flow.raft import FlowRaft
            objType = FlowRaft

    elif type == "semantic":
        if method == "safeuav-keras":
            from .semantic.safeuav_keras import SSegSafeUAVKeras
            objType = SSegSafeUAVKeras
        elif method == "safeuav":
            from .semantic.safeuav import SSegSafeUAV
            objType = SSegSafeUAV

    elif type == "normals":
        if method == "depth-svd":
            from .normals.depth_svd import DepthNormalsSVD
            objType = DepthNormalsSVD

    assert objType is not None, f"Unknown type: {type}, method: {method}"
    return objType