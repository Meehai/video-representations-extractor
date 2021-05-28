from typing import Dict
from .representation import Representation

def getRepresentation(item) -> Representation:
    print("[getRepresentation] Instantiating Method='%s'..." % item["method"])
    obj = None
    if item["method"] == "rgb":
        from .rgb import RGB
        obj = RGB()
    elif item["method"] == "hsv":
        from .hsv import HSV
        obj = HSV()
    elif item["method"] == "python-halftone":
        from .python_halftone import Halftone
        obj = Halftone(**item["parameters"])
    elif item["method"] == "dexined":
        from .dexined import DexiNed
        obj = DexiNed()
    elif item["method"] == "jiaw":
        from .depth_jiaw import DepthJiaw
        obj = DepthJiaw(**item["parameters"])
    elif item["method"] == "dpt":
        from .depth_dpt import DepthDpt
        obj = DepthDpt(**item["parameters"])
    elif item["method"] == "rife":
        from .flow_rife import FlowRife
        obj = FlowRife(**item["parameters"])
    else:
        assert False, "Unknown method: %s/%s" % (group, method)
    
    return obj