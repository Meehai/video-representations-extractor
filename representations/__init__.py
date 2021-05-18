from typing import Dict
from .representation import Representation

def getRepresentation(Type:str, args:Dict) -> Representation:
    obj = None
    if Type == "rgb":
        from .rgb import RGB
        obj = RGB()
    elif Type == "hsv":
        from .hsv import HSV
        obj = HSV()
    elif Type == "halftone" and args["method"] == "python-halftone":
        from .python_halftone import Halftone
        obj = Halftone(**args["parameters"])
    else:
        assert False, "Unknown representation: %s or method: %s" % (Type, args["method"])
    
    return obj