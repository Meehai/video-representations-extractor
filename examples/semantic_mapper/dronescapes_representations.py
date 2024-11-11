"""dronescapes representations specific for Dronescapes Dataset"""
from vre.representations import Representation, TaskMapper, IORepresentationMixin, NpIORepresentation
from vre.utils import semantic_mapper, colorize_semantic_segmentation
from vre.representations.cv_representations import (
    ColorRepresentation, RGBRepresentation, HSVRepresentation, DepthRepresentation,
    OpticalFlowRepresentation, EdgesRepresentation, NormalsRepresentation, SemanticRepresentation)
import numpy as np
import torch as tr
from torch.nn import functional as F
from overrides import overrides

dronescapes_color_map = [[0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 255, 255],
                         [255, 0, 0], [0, 0, 255], [0, 255, 255], [127, 127, 63]]
dronescapes_classes = ["land", "forest", "residential", "road", "little-objects", "water", "sky", "hill"]
coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush", "banner", "blanket", "bridge", "cardboard",
                "counter", "curtain", "door-stuff", "floor-wood", "flower", "fruit", "gravel", "house", "light",
                "mirror-stuff", "net", "pillow", "platform", "playingfield", "railroad", "river", "road", "roof",
                "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick", "wall-stone", "wall-tile",
                "wall-wood", "water-other", "window-blind", "window-other", "tree-merged", "fence-merged",
                "ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged", "floor-other-merged",
                "pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged",
                "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged", "rug-merged"]
coco_color_map = [[220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100],
                  [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30],
                  [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157],
                  [110, 76, 0], [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240], [0, 125, 92],
                  [209, 0, 151], [188, 208, 182], [0, 220, 176], [255, 99, 164], [92, 0, 73], [133, 129, 255],
                  [78, 180, 255], [0, 228, 0], [174, 255, 243], [45, 89, 255], [134, 134, 103], [145, 148, 174],
                  [255, 208, 186], [197, 226, 255], [171, 134, 1], [109, 63, 54], [207, 138, 255], [151, 0, 95],
                  [9, 80, 61], [84, 105, 51], [74, 65, 105], [166, 196, 102], [208, 195, 210], [255, 109, 65],
                  [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0], [227, 255, 205], [147, 186, 208],
                  [153, 69, 1], [3, 95, 161], [163, 255, 0], [119, 0, 170], [0, 182, 199], [0, 165, 120],
                  [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133], [166, 74, 118], [219, 142, 185],
                  [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106], [142, 108, 45],
                  [196, 172, 0], [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122], [191, 162, 208],
                  [255, 255, 128], [147, 211, 203], [150, 100, 100], [168, 171, 172], [146, 112, 198],
                  [210, 170, 100], [92, 136, 89], [218, 88, 184], [241, 129, 0], [217, 17, 255], [124, 74, 181],
                  [70, 70, 70], [255, 228, 255], [154, 208, 0], [193, 0, 92], [76, 91, 113], [255, 180, 195],
                  [106, 154, 176], [230, 150, 140], [60, 143, 255], [128, 64, 128], [92, 82, 55], [254, 212, 124],
                  [73, 77, 174], [255, 160, 98], [255, 255, 255], [104, 84, 109], [169, 164, 131], [225, 199, 255],
                  [137, 54, 74], [135, 158, 223], [7, 246, 231], [107, 255, 200], [58, 41, 149], [183, 121, 142],
                  [255, 73, 97], [107, 142, 35], [190, 153, 153], [146, 139, 141], [70, 130, 180], [134, 199, 156],
                  [209, 226, 140], [96, 36, 108], [96, 96, 96], [64, 170, 64], [152, 251, 152], [208, 229, 228],
                  [206, 186, 171], [152, 161, 64], [116, 112, 0], [0, 114, 143], [102, 102, 156], [250, 141, 255]]
mapillary_classes = ["Bird", "Ground Animal", "Curb", "Fence", "Guard Rail", "Barrier", "Wall", "Bike Lane",
                     "Crosswalk - Plain", "Curb Cut", "Parking", "Pedestrian Area", "Rail Track", "Road",
                     "Service Lane", "Sidewalk", "Bridge", "Building", "Tunnel", "Person", "Bicyclist",
                     "Motorcyclist", "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General",
                     "Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation", "Water", "Banner", "Bench",
                     "Bike Rack", "Billboard", "Catch Basin", "CCTV Camera", "Fire Hydrant", "Junction Box",
                     "Mailbox", "Manhole", "Phone Booth", "Pothole", "Street Light", "Pole", "Traffic Sign Frame",
                     "Utility Pole", "Traffic Light", "Traffic Sign (Back)", "Traffic Sign (Front)", "Trash Can",
                     "Bicycle", "Boat", "Bus", "Car", "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer",
                     "Truck", "Wheeled Slow", "Car Mount", "Ego Vehicle"]
mapillary_color_map = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153], [180, 165, 180], [90, 120, 150],
                       [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
                       [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232], [150, 100, 100], [70, 70, 70],
                       [150, 120, 90], [220, 20, 60], [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
                       [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180], [190, 255, 255], [152, 251, 152],
                       [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
                       [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40], [33, 33, 33], [100, 128, 160],
                       [142, 0, 0], [70, 100, 150], [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
                       [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20], [119, 11, 32], [150, 0, 255],
                       [0, 60, 100], [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
                       [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10]]


class SemanticMask2FormerMapillaryConvertedPaper(TaskMapper, NpIORepresentation):
    def __init__(self, name: str, dependencies: list[Representation]):
        TaskMapper.__init__(self, name=name, n_channels=8, dependencies=dependencies)
        NpIORepresentation.__init__(self)
        self.mapping = {
            "land": ["Terrain", "Sand", "Snow"],
            "forest": ["Vegetation"],
            "residential": ["Building", "Utility Pole", "Pole", "Fence", "Wall", "Manhole", "Street Light", "Curb",
                            "Guard Rail", "Caravan", "Junction Box", "Traffic Sign (Front)", "Billboard", "Banner",
                            "Mailbox", "Traffic Sign (Back)", "Bench", "Fire Hydrant", "Trash Can", "CCTV Camera",
                            "Traffic Light", "Barrier", "Rail Track", "Phone Booth", "Curb Cut", "Traffic Sign Frame",
                            "Bike Rack"],
            "road": ["Road", "Lane Marking - General", "Sidewalk", "Bridge", "Other Vehicle", "Motorcyclist", "Pothole",
                     "Catch Basin", "Car Mount", "Tunnel", "Parking", "Service Lane", "Lane Marking - Crosswalk",
                     "Pedestrian Area", "On Rails", "Bike Lane", "Crosswalk - Plain"],
            "little-objects": ["Car", "Person", "Truck", "Boat", "Wheeled Slow", "Trailer", "Ground Animal", "Bicycle",
                               "Motorcycle", "Bird", "Bus", "Ego Vehicle", "Bicyclist", "Other Rider"],
            "water": ["Water"],
            "sky": ["Sky"],
            "hill": ["Mountain"]
        }
        self.color_map = dronescapes_color_map
        self.original_classes = mapillary_classes
        self.classes = list(self.mapping.keys())
        self.n_classes = len(self.classes)

    @overrides
    def make_images(self) -> np.ndarray:
        res = [colorize_semantic_segmentation(item.argmax(-1).astype(int), self.classes, self.color_map,
                                              original_rgb=None, font_size_scale=2) for item in self.data.output]
        return np.array(res)

    @overrides
    def merge_fn(self, dep_data: list[np.ndarray]) -> np.ndarray:
        m2f_mapillary = dep_data[0].argmax(-1)
        m2f_mapillary_converted = semantic_mapper(m2f_mapillary, self.mapping, self.original_classes)
        return self.from_disk_fmt(m2f_mapillary_converted)

    @overrides
    def to_disk_fmt(self, memory_data: np.ndarray) -> np.ndarray:
        return memory_data.argmax(-1).astype(np.uint8)

    @overrides
    def from_disk_fmt(self, disk_data: np.ndarray) -> np.ndarray:
        return np.eye(self.n_classes)[disk_data.astype(int)]

class SemanticMask2FormerCOCOConverted(TaskMapper, NpIORepresentation):
    def __init__(self, name: str, dependencies: list[Representation]):
        TaskMapper.__init__(self, name=name, n_channels=8, dependencies=dependencies)
        NpIORepresentation.__init__(self)
        self.mapping = {
            "land": ["grass-merged", "dirt-merged", "sand", "gravel", "flower", "playingfield", "snow", "platform"],
            "forest": ["tree-merged"],
            "residential": ["building-other-merged", "house", "roof", "fence-merged", "wall-other-merged", "wall-brick",
                            "rock-merged", "tent", "bridge", "bench", "window-other", "fire hydrant", "traffic light",
                            "umbrella", "wall-stone", "clock", "chair", "sports ball", "floor-other-merged",
                            "floor-wood", "stop sign", "door-stuff", "banner", "light", "net", "surfboard", "frisbee",
                            "rug-merged", "potted plant", "parking meter"],
            "road": ["road", "railroad", "pavement-merged", "stairs"],
            "little-objects": ["truck", "car", "boat", "horse", "person", "train", "elephant", "bus", "bird", "sheep",
                               "cow", "motorcycle", "dog", "bicycle", "airplane", "kite"],
            "water": ["river", "water-other", "sea"],
            "sky": ["sky-other-merged"],
            "hill": ["mountain-merged"]
        }
        self.color_map = dronescapes_color_map
        self.original_classes = coco_classes
        self.classes = list(self.mapping.keys())
        self.n_classes = len(self.classes)
        self.output_dtype = "uint8"

    @overrides
    def make_images(self) -> np.ndarray:
        res = [colorize_semantic_segmentation(item.argmax(-1).astype(int), self.classes, self.color_map,
                                              original_rgb=None, font_size_scale=2) for item in self.data.output]
        return np.array(res)

    @overrides
    def merge_fn(self, dep_data: list[np.ndarray]) -> np.ndarray:
        m2f_mapillary = dep_data[0].argmax(-1)
        m2f_mapillary_converted = semantic_mapper(m2f_mapillary, self.mapping, self.original_classes)
        return self.from_disk_fmt(m2f_mapillary_converted)

    @overrides
    def to_disk_fmt(self, memory_data: np.ndarray) -> np.ndarray:
        return memory_data.argmax(-1).astype(np.uint8)

    @overrides
    def from_disk_fmt(self, disk_data: np.ndarray) -> np.ndarray:
        return np.eye(self.n_classes)[disk_data.astype(int)]

_tasks: list[Representation | IORepresentationMixin] = [ # some pre-baked representations
    # Simple color ones
    rgb := RGBRepresentation("rgb"),
    HSVRepresentation("hsv", dependencies=[rgb]),
    ColorRepresentation("softseg_gb"),
    # Edges
    EdgesRepresentation("edges_dexined"),
    EdgesRepresentation("edges_gb"),
    # Depth
    DepthRepresentation("depth_dpt", min_depth=0, max_depth=0.999),
    DepthRepresentation("depth_sfm_manual202204", min_depth=0, max_depth=300),
    DepthRepresentation("depth_ufo", min_depth=0, max_depth=1),
    DepthRepresentation("depth_marigold", min_depth=0, max_depth=1),
    # Normals
    NormalsRepresentation("normals_sfm_manual202204"),
    NormalsRepresentation("normals_svd(depth_marigold)"),
    # Optical Flow
    OpticalFlowRepresentation("opticalflow_rife"),
    # Semantic Segmentation
    SemanticRepresentation("semantic_segprop8", classes=dronescapes_classes, color_map=dronescapes_color_map),
    m2f_coco := SemanticRepresentation("semantic_mask2former_coco_47429163_0", classes=coco_classes,
                                       color_map=coco_color_map),
    m2f_mapillary := SemanticRepresentation("semantic_mask2former_mapillary_49189528_0", classes=mapillary_classes,
                                            color_map=mapillary_color_map),
    # Semantic mapped representations
    SemanticMask2FormerMapillaryConvertedPaper("semantic_mask2former_swin_mapillary_converted", [m2f_mapillary]),
    SemanticMask2FormerCOCOConverted("semantic_mask2former_swin_coco_converted", [m2f_coco]),
]
dronescapes_task_types: dict[str, Representation | IORepresentationMixin] = {task.name: task for task in _tasks}
