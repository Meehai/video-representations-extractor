#!/usr/bin/env python3
"""semantic_mapper.py -- primivites for new tasks based on existing CV/dronescapes tasks"""
from overrides import overrides
from pathlib import Path
from functools import reduce
from pprint import pprint
import numpy as np
import torch as tr

from vre.utils import (semantic_mapper, colorize_semantic_segmentation, DiskData, MemoryData, ReprOut, reorder_dict,
                       collage_fn, image_add_title)
from vre.readers.multitask_dataset import MultiTaskDataset, MultiTaskItem
from vre.representations import TaskMapper, NpIORepresentation, Representation, build_representations_from_cfg
from vre.representations.semantic_segmentation import SemanticRepresentation
from vre.representations.cv_representations import DepthRepresentation, NormalsRepresentation

def plot_one(data: MultiTaskItem, title: str, order: list[str] | None,
             name_to_task: dict[str, Representation]) -> np.ndarray:
    """simple plot function: plot_one(reader[0][0], reader[0][1], None, reader.name_to_task)"""
    def vre_plot_fn(rgb: tr.Tensor, x: tr.Tensor, node: Representation) -> np.ndarray:
        node.data = ReprOut(rgb.cpu().detach().numpy()[None], MemoryData(x.cpu().detach().numpy()[None]), [0])
        return node.make_images(node.data)[0]
    img_data = {k: vre_plot_fn(data["rgb"], v, name_to_task[k]) for k, v in data.items()}
    img_data = reorder_dict(img_data, order) if order is not None else img_data
    titles = [title if len(title) < 40 else f"{title[0:19]}..{title[-19:]}" for title in img_data]
    collage = collage_fn(list(img_data.values()), titles=titles, size_px=40)
    collage = image_add_title(collage, title, size_px=55, top_padding=110)
    return collage

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

m2f_coco = SemanticRepresentation("semantic_mask2former_coco_47429163_0", classes=coco_classes,
                                    color_map=coco_color_map)
m2f_mapillary = SemanticRepresentation("semantic_mask2former_mapillary_49189528_0", classes=mapillary_classes,
                                        color_map=mapillary_color_map)
marigold = DepthRepresentation("depth_marigold", min_depth=0, max_depth=1)
normals_svd_marigold = NormalsRepresentation("normals_svd(depth_marigold)")

class SemanticMask2FormerMapillaryConvertedPaper(TaskMapper, NpIORepresentation):
    def __init__(self, name: str, dependencies: list[Representation]):
        assert len(dependencies) == 1 and isinstance(dependencies[0], SemanticRepresentation), dependencies
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
        self.color_map = [[0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 255, 255],
                          [255, 0, 0], [0, 0, 255], [0, 255, 255], [127, 127, 63]]
        self.original_classes = dependencies[0].classes
        assert set(reduce(lambda x, y: x + y, self.mapping.values(), [])) == set(self.original_classes)
        self.classes = list(self.mapping.keys())
        self.n_classes = len(self.classes)
        self.output_dtype = "uint8"

    @property
    @overrides
    def n_channels(self) -> int:
        return self.n_classes

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        return colorize_semantic_segmentation(data.output.argmax(-1), self.classes, self.color_map)

    @overrides
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        m2f_mapillary = self.dependencies[0].to_argmaxed_representation(dep_data[0])
        m2f_mapillary_converted = semantic_mapper(m2f_mapillary, self.mapping, self.original_classes)
        return self.disk_to_memory_fmt(m2f_mapillary_converted)

    @overrides
    def memory_to_disk_fmt(self, memory_data: MemoryData) -> DiskData:
        return memory_data.argmax(-1).astype(np.uint8)

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        return MemoryData(np.eye(self.n_classes)[disk_data.astype(int)])

class SemanticMask2FormerCOCOConverted(TaskMapper, NpIORepresentation):
    def __init__(self, name: str, dependencies: list[SemanticRepresentation]):
        assert len(dependencies) == 1 and isinstance(dependencies[0], SemanticRepresentation), dependencies
        TaskMapper.__init__(self, name=name, n_channels=8, dependencies=dependencies)
        NpIORepresentation.__init__(self)
        self.mapping = {
            "land": ["grass-merged", "dirt-merged", "sand", "gravel", "flower", "playingfield", "snow", "platform"],
            "forest": ["tree-merged"],
            "residential": ["building-other-merged", "house", "roof", "fence-merged", "wall-other-merged", "wall-brick",
                            "rock-merged", "tent", "bridge", "bench", "window-other", "fire hydrant", "traffic light",
                            "umbrella", "wall-stone", "clock", "chair", "sports ball", "floor-other-merged",
                            "floor-wood", "stop sign", "door-stuff", "banner", "light", "net", "surfboard", "frisbee",
                            "rug-merged", "potted plant", "parking meter", "tennis racket", "sink", "hair drier",
                            "food-other-merged", "curtain", "mirror-stuff", "baseball glove", "baseball bat", "zebra",
                            "spoon", "towel", "donut", "apple", "handbag", "couch", "orange", "wall-wood",
                            "window-blind", "pizza", "cabinet-merged", "skateboard", "remote", "bottle", "bed",
                            "table-merged", "backpack", "bear", "wall-tile", "cup", "scissors", "ceiling-merged",
                            "oven", "cell phone", "microwave", "toaster", "carrot", "fork", "giraffe", "paper-merged",
                            "cat", "book", "sandwich", "wine glass", "pillow", "blanket", "tie", "bowl", "snowboard",
                            "vase", "toothbrush", "toilet", "dining table", "laptop", "tv", "cardboard", "keyboard",
                            "hot dog", "cake", "knife", "suitcase", "refrigerator", "fruit", "shelf", "counter", "skis",
                            "banana", "teddy bear", "broccoli", "mouse"],
            "road": ["road", "railroad", "pavement-merged", "stairs"],
            "little-objects": ["truck", "car", "boat", "horse", "person", "train", "elephant", "bus", "bird", "sheep",
                               "cow", "motorcycle", "dog", "bicycle", "airplane", "kite"],
            "water": ["river", "water-other", "sea"],
            "sky": ["sky-other-merged"],
            "hill": ["mountain-merged"]
        }
        self.color_map = [[0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 255, 255],
                          [255, 0, 0], [0, 0, 255], [0, 255, 255], [127, 127, 63]]
        self.original_classes = dependencies[0].classes
        assert set(reduce(lambda x, y: x + y, self.mapping.values(), [])) == set(self.original_classes)
        self.classes = list(self.mapping.keys())
        self.n_classes = len(self.classes)
        self.output_dtype = "uint8"

    @property
    @overrides
    def n_channels(self) -> int:
        return self.n_classes

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        return colorize_semantic_segmentation(data.output.argmax(-1), self.classes, self.color_map)

    @overrides
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        m2f_mapillary = self.dependencies[0].to_argmaxed_representation(dep_data[0])
        m2f_mapillary_converted = semantic_mapper(m2f_mapillary, self.mapping, self.original_classes)
        res = self.disk_to_memory_fmt(m2f_mapillary_converted)
        return res

    @overrides
    def memory_to_disk_fmt(self, memory_data: MemoryData) -> DiskData:
        return memory_data.argmax(-1).astype(np.uint8)

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        return MemoryData(np.eye(self.n_classes)[disk_data.astype(int)])

class BinaryMapper(TaskMapper, NpIORepresentation):
    """
    Note for future self: this is never generic enough to be in VRE -- we'll keep it in this separate code only
    TaskMapper is the only high level interface that makes sense, so we should focus on keeping that generic and easy.
    """
    def __init__(self, name: str, dependencies: list[Representation], mapping: list[dict[str, list]],
                 mode: str = "all_agree", load_mode: str = "binary"):
        TaskMapper.__init__(self, name=name, dependencies=dependencies, n_channels=2)
        NpIORepresentation.__init__(self)
        assert mode in ("all_agree", "at_least_one"), mode
        assert load_mode in ("one_hot", "binary")
        assert len(mapping[0]) == 2, mapping
        assert len(mapping) == len(dependencies), (len(mapping), len(dependencies))
        assert all(mapping[0].keys() == m.keys() for m in mapping), [m.keys() for m in mapping]
        self.original_classes = [dep.classes for dep in dependencies]
        self.mapping = mapping
        self.mode = mode
        self.load_mode = load_mode
        self.classes = list(mapping[0].keys())
        self.n_classes = len(self.classes)
        self.color_map = [[255, 255, 255], [0, 0, 0]]
        self.output_dtype = "bool"

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        x = data.output.argmax(-1) if self.load_mode == "one_hot" else (data.output > 0.5).astype(int)
        x = x[..., 0] if x.shape[-1] == 1 else x
        return colorize_semantic_segmentation(x, self.classes, self.color_map)

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        y = np.eye(2)[disk_data.astype(int)] if self.load_mode == "one_hot" else disk_data
        return MemoryData(y.astype(np.float32))

    @overrides
    def memory_to_disk_fmt(self, memory_data: MemoryData) -> DiskData:
        return memory_data.argmax(-1).astype(bool) if self.load_mode == "one_hot" else memory_data.astype(bool)

    @overrides
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        dep_data_argmaxed = []
        for dep, data in zip(self.dependencies, dep_data):
            assert isinstance(dep, SemanticRepresentation), type(dep)
            dep_data_argmaxed.append(dep.to_argmaxed_representation(data))
        dep_data_converted = [semantic_mapper(x, mapping, oc)
                              for x, mapping, oc in zip(dep_data_argmaxed, self.mapping, self.original_classes)]
        res_argmax = sum(dep_data_converted) > (0 if self.mode == "all_agree" else 1)
        return self.disk_to_memory_fmt(res_argmax)

class BuildingsFromM2FDepth(BinaryMapper, NpIORepresentation):
    def __init__(self, name: str, original_classes: tuple[list[str], list[str]], load_mode: str = "binary"):
        buildings_mapping = [
            {
                "buildings": (cls := ["Building", "Utility Pole", "Pole", "Fence", "Wall"]),
                "others": [x for x in mapillary_classes if x not in cls],
            },
            {
                "buildings": (cls := ["building-other-merged", "house", "roof"]),
                "others": [x for x in coco_classes if x not in cls]
            }
        ]

        dependencies = [m2f_mapillary, m2f_coco, marigold]
        assert isinstance(m2f_mapillary, SemanticRepresentation) and isinstance(m2f_coco, SemanticRepresentation)
        TaskMapper.__init__(self, name=name, dependencies=dependencies, n_channels=2)
        NpIORepresentation.__init__(self)
        self.color_map = [[255, 255, 255], [0, 0, 0]]
        self.original_classes = original_classes
        self.mapping = buildings_mapping
        self.classes = list(buildings_mapping[0].keys())
        self.n_classes = len(self.classes)
        self.load_mode = load_mode

    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        m2f_mapillary = self.dependencies[0].to_argmaxed_representation(dep_data[0])
        m2f_coco = self.dependencies[1].to_argmaxed_representation(dep_data[1])
        depth = dep_data[2].squeeze()
        m2f_mapillary_converted = semantic_mapper(m2f_mapillary, self.mapping[0], self.original_classes[0])
        m2f_coco_converted = semantic_mapper(m2f_coco, self.mapping[1], self.original_classes[1])
        thr = 0.3 # np.percentile(depth.numpy(), 0.8)
        combined = (m2f_mapillary_converted + m2f_coco_converted + (depth > thr)) != 0
        return self.disk_to_memory_fmt(combined)

class SafeLandingAreas(BinaryMapper, NpIORepresentation):
    def __init__(self, name: str, original_classes: tuple[list[str], list[str]], include_semantics: bool = False,
                 sky_water: BinaryMapper | None = None, load_mode: str = "binary"):
        self.include_semantics = include_semantics
        dependencies = [m2f_mapillary, m2f_coco, marigold, normals_svd_marigold]
        TaskMapper.__init__(self, name, dependencies=dependencies, n_channels=2)
        NpIORepresentation.__init__(self)
        self.color_map = [[0, 255, 0], [255, 0, 0]]
        self.original_classes = original_classes
        self.classes = ["safe-landing", "unsafe-landing"]
        self.n_classes = len(self.classes)
        self.sky_water = sky_water
        self.load_mode = load_mode

    @overrides
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        normals, depth = dep_data[3], dep_data[2].squeeze()
        v1, v2, v3 = normals.transpose(2, 0, 1)
        where_safe = (v2 > 0.8) * ((v1 + v3) < 1.2)
        if self.include_semantics:
            sw = self.sky_water.merge_fn(dep_data)
            sw = sw.argmax(-1) if self.sky_water.load_mode == "one_hot" else sw
            where_safe = (where_safe * sw * (depth < 0.9)).astype(bool)
        return self.disk_to_memory_fmt(~where_safe)

def get_new_semantic_mapped_tasks(tasks_subset: list[str] | None = None) -> dict[str, TaskMapper]:
    """The exported function for VRE!"""
    buildings_mapping = [
        {
            "buildings": (cls := ["Building", "Utility Pole", "Pole", "Fence", "Wall"]),
            "others": [x for x in mapillary_classes if x not in cls],
        },
        {
            "buildings": (cls := ["building-other-merged", "house", "roof"]),
            "others": [x for x in coco_classes if x not in cls]
        }
    ]

    living_mapping = [
        {
            "living": (cls := ["Person", "Bicyclist", "Motorcyclist", "Other Rider", "Bird", "Ground Animal"]),
            "others": [c for c in mapillary_classes if c not in cls],
        },
        {
            "living": (cls := ["person", "bird", "cat", "dog", "horse", "sheep", "cow",
                            "elephant", "bear", "zebra", "giraffe"]),
            "others": [c for c in coco_classes if c not in cls],
        }
    ]

    sky_and_water_mapping = [
        {
            "sky-and-water": (cls := ["Sky", "Water"]),
            "others": [c for c in mapillary_classes if c not in cls],
        },
        {
            "sky-and-water": (cls := ["sky-other-merged", "water-other"]),
            "others": [c for c in coco_classes if c not in cls],
        },
    ]

    transportation_mapping = [
        {
            "transportation": (cls := ["Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Rail Track",
                                       "Road", "Service Lane", "Sidewalk", "Bridge", "Tunnel", "Bicyclist",
                                       "Motorcyclist",
                                       "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General",
                                       "Traffic Light",
                                       "Traffic Sign (Back)", "Traffic Sign (Front)", "Bicycle", "Boat", "Bus", "Car",
                                       "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck",
                                       "Wheeled Slow", "Car Mount", "Ego Vehicle"]),
            "others": [c for c in mapillary_classes if c not in cls]
        },
        {
            "transportation": (cls := ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]),
            "others": [c for c in coco_classes if c not in cls]
        }
    ]

    containing_mapping = [
        {
            "containing": (cls := [
                "Terrain", "Sand", "Mountain", "Road", "Sidewalk", "Pedestrian Area", "Rail Track", "Parking",
                "Service Lane", "Bridge", "Water", "Vegetation", "Curb", "Fence", "Wall", "Guard Rail",
                "Barrier", "Curb Cut", "Snow"
            ]),
            "contained": [c for c in mapillary_classes if c not in cls],  # Buildings and constructions will be here
        },
        {
            "containing": (cls := [
                "floor-wood", "floor-other-merged", "pavement-merged", "mountain-merged", "sand", "road",
                "sea", "river", "railroad", "platform", "grass-merged", "snow", "stairs", "tent"
            ]),
            "contained": [c for c in coco_classes if c not in cls],  # Buildings and constructions will be here
        }
    ]

    vegetation_mapping = [
        {
            "vegetation": (cls := ["Mountain", "Sand", "Sky", "Snow", "Terrain", "Vegetation"]),
            "others": [x for x in mapillary_classes if x not in cls],
        },
        {
            "vegetation": (cls := ["tree-merged", "grass-merged", "dirt-merged", "flower", "potted plant", "river",
                                "sea", "water-other", "mountain-merged", "rock-merged"]),
            "others": [x for x in coco_classes if x not in cls],
        }
    ]

    available_tasks: list[TaskMapper] = [
        SemanticMask2FormerMapillaryConvertedPaper("semantic_mask2former_swin_mapillary_converted", [m2f_mapillary]),
        SemanticMask2FormerCOCOConverted("semantic_mask2former_swin_coco_converted", [m2f_coco]),
        BinaryMapper("buildings", [m2f_mapillary, m2f_coco], buildings_mapping),
        BinaryMapper("living-vs-non-living", [m2f_mapillary, m2f_coco], living_mapping),
        sky_water := BinaryMapper("sky-and-water", [m2f_mapillary, m2f_coco], sky_and_water_mapping,
                                  mode="at_least_one"),
        BinaryMapper("transportation", [m2f_mapillary, m2f_coco], transportation_mapping, mode="at_least_one"),
        BinaryMapper("containing", [m2f_mapillary, m2f_coco], containing_mapping),
        BinaryMapper("vegetation", [m2f_mapillary, m2f_coco], vegetation_mapping),
        BuildingsFromM2FDepth("buildings(nearby)", [mapillary_classes, coco_classes]),
        SafeLandingAreas("safe-landing-no-sseg", [mapillary_classes, coco_classes]),
        SafeLandingAreas("safe-landing-semantics", [mapillary_classes, coco_classes],
                         include_semantics=True, sky_water=sky_water),
    ]
    if tasks_subset is None:
        return {t.name: t for t in available_tasks}
    return {t.name: t for t in available_tasks if t.name in tasks_subset}

if __name__ == "__main__":
    cfg_path = Path.cwd() / "cfg.yaml"
    data_path = Path.cwd() / "data"
    vre_dir = data_path

    task_names = ["rgb", "depth_marigold", "normals_svd(depth_marigold)", "opticalflow_rife",
                "semantic_mask2former_coco_47429163_0", "semantic_mask2former_mapillary_49189528_0"]
    order = ["rgb", "semantic_mask2former_mapillary_49189528_0", "semantic_mask2former_coco_47429163_0",
                "depth_marigold", "normals_svd(depth_marigold)"]

    representations = build_representations_from_cfg(cfg_path)
    reader = MultiTaskDataset(vre_dir, task_names=task_names,
                            task_types=representations, handle_missing_data="fill_nan",
                            normalization="min_max", cache_task_stats=True, batch_size_stats=100)
    orig_task_names = list(reader.task_types.keys())

    new_tasks = get_new_semantic_mapped_tasks()
    for task_name in reader.task_names:
        if task_name not in orig_task_names:
            reader.remove_task(task_name)
    for new_task in new_tasks.values():
        reader.add_task(new_task, overwrite=True)

    print("== Random loaded item ==")
    ixs = np.random.permutation(range(len(reader))).tolist()
    for ix in ixs:
        data, name = reader[ix]
        pprint(data)
        print(plot_one(data, title=name, order=order, name_to_task=reader.name_to_task).shape)
        break
