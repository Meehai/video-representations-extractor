from .representation import Representation

def getRepresentation(method:str) -> Representation:
    print("[getRepresentation] Instantiating Method='%s'..." % method)
    objType = None
    if method == "rgb":
        from .rgb import RGB
        objType = RGB
    elif method == "hsv":
        from .hsv import HSV
        objType = HSV
    elif method == "python-halftone":
        from .python_halftone import Halftone
        objType = Halftone
    elif method == "dexined":
        from .dexined import DexiNed
        objType = DexiNed
    elif method == "dpt":
        from .depth_dpt import DepthDpt
        objType = DepthDpt
    elif method == "rife":
        from .flow_rife import FlowRife
        objType = FlowRife
    elif method == "raft":
        from .flow_raft import FlowRaft
        objType = FlowRaft
    elif method == "depth-odo-flow":
        from .depth_odo_flow import DepthOdoFlow
        objType = DepthOdoFlow
    elif method == "canny":
        from .canny import Canny
        objType = Canny
    elif method == "kmeans":
        from .kmeans import KMeans
        objType = KMeans
    # Complex pretrained.
    elif method == "semantic-safeuav-keras":
        from .sseg_safeuav_keras import SSegSafeUAVKeras
        objType = SSegSafeUAVKeras
    elif method == "semantic-safeuav":
        from .sseg_safeuav import SSegSafeUAV
        objType = SSegSafeUAV
    elif method == "depth-dispresnet":
        from .depth_dispresnet import DepthDispResNet
        objType = DepthDispResNet
    else:
        assert False, "Unknown method: %s" % method
    return objType