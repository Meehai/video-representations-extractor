from typing import Dict
from .representation import Representation

def getRepresentation(method) -> Representation:
    print("[getRepresentation] Instantiating Method='%s'..." % method)
    obj = None
    if method == "rgb":
        from .rgb import RGB
        obj = RGB
    elif method == "hsv":
        from .hsv import HSV
        obj = HSV
    elif method == "python-halftone":
        from .python_halftone import Halftone
        obj = Halftone
    elif method == "dexined":
        from .dexined import DexiNed
        obj = DexiNed
    elif method == "jiaw":
        from .depth_jiaw import DepthJiaw
        obj = DepthJiaw
    elif method == "dpt":
        from .depth_dpt import DepthDpt
        obj = DepthDpt
    elif method == "rife":
        from .flow_rife import FlowRife
        obj = FlowRife
    elif method == "semantic-safeuav-keras":
        from .sseg_safeuav_keras import SSegSafeUAVKeras
        obj = SSegSafeUAVKeras
    elif method == "canny":
        from .canny import Canny
        obj = Canny
    elif method == "kmeans":
        from .kmeans import KMeans
        obj = KMeans
    else:
        assert False, "Unknown method: %s" % method
    
    return obj