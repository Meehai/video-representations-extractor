"""init file for vre default repository"""
from typing import Type
from vre.representations import Representation

# pylint: disable=import-outside-toplevel
def get_vre_repository() -> dict[str, Type[Representation]]:
    """the built-in repository for VRE"""
    from .color.rgb import RGB
    from .color.hsv import HSV
    from .soft_segmentation.halftone import Halftone
    from .soft_segmentation.generalized_boundaries import GeneralizedBoundaries
    from .soft_segmentation.fastsam import FastSam
    from .edges.canny import Canny
    from .edges.dexined import DexiNed
    from .depth.dpt import DepthDpt
    from .depth.marigold import Marigold
    from .optical_flow.rife import FlowRife
    from .optical_flow.raft import FlowRaft
    from .semantic_segmentation.safeuav import SafeUAV
    from .semantic_segmentation.mask2former import Mask2Former
    from .normals.depth_svd import DepthNormalsSVD

    return {
        "color/rgb": RGB,
        "color/hsv": HSV,
        "soft-segmentation/python-halftone": Halftone,
        "soft-segmentation/generalized-boundaries": GeneralizedBoundaries,
        "soft-segmentation/fastsam": FastSam,
        "edges/canny": Canny,
        "edges/dexined": DexiNed,
        "depth/dpt": DepthDpt,
        "depth/marigold": Marigold,
        "semantic-segmentation/safeuav": SafeUAV,
        "semantic-segmentation/mask2former": Mask2Former,
        "optical-flow/rife": FlowRife,
        "optical-flow/raft": FlowRaft,
        "normals/depth-svd": DepthNormalsSVD,
    }
