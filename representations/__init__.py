from typing import Dict
from .representation import Representation

def getRepresentation(Type:str, args:Dict):
    if Type == "halftone" and args["method"] == "python-halftone":
        from .python_halftone import Halftone
        obj = Halftone(**args["parameters"])
        return obj
    else:
        assert False, "Unknown representation: %s or method: %s" % (Type, args["method"])
    print(Type, args)
    pass