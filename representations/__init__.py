from typing import Dict
from .representation import Representation

def getRepresentation(item) -> Representation:
    group, method = item["group"], item["method"]
    print("[getRepresentation] Instantiating Group='%s' Method='%s'..." % (group, method))
    obj = None
    if group == "rgb":
        assert method == "rgb", "Unknown method: %s/%s" % (group, method)
        from .rgb import RGB
        obj = RGB()
    elif group == "hsv":
        assert method == "hsv", "Unknown method: %s/%s" % (group, method)
        from .hsv import HSV
        obj = HSV()
    elif group == "halftone":
        assert method in ("python-halftone", ), "Unknown method: %s/%s" % (group, method)
        if method == "python-halftone":
            from .python_halftone import Halftone
            obj = Halftone(**item["parameters"])
    elif group == "edgeDetection":
        assert method in ("dexined", ), "Unknown method: %s/%s" % (group, method)
        if method == "dexined":
            from .dexined import DexiNed
            obj = DexiNed()
    elif group == "depthEstimation":
        assert method in ("jiaw", "dpt"), "Unknown method: %s/%s" % (group, method)
        if method == "jiaw":
            from .depth_jiaw import DepthJiaw
            obj = DepthJiaw(**item["parameters"])
        elif method == "dpt":
            from .depth_dpt import DepthDpt
            obj = DepthDpt(**item["parameters"])
    elif group == "opticalFlow":
        assert method in ("rife", ), "Unknown method: %s/%s" % (group, method)
        if method == "rife":
            from .flow_rife import FlowRife
            obj = FlowRife(**item["parameters"])
    else:
        assert False, "Unknown method: %s/%s" % (group, method)
    
    return obj