#!/usr/bin/env python3
"""semantic_mapper.py -- primivites for new tasks based on existing CV/dronescapes tasks"""
from overrides import overrides
from pathlib import Path
from functools import reduce
from pprint import pprint
import numpy as np
import torch as tr

from vre.utils import (semantic_mapper, colorize_semantic_segmentation, DiskData, MemoryData,
                       ReprOut, reorder_dict, collage_fn, image_add_title, lo, image_write)
from vre.logger import vre_logger as logger
from vre.readers.multitask_dataset import MultiTaskDataset, MultiTaskItem
from vre.representations import TaskMapper, NpIORepresentation, Representation, build_representations_from_cfg
from vre_repository import get_vre_repository
from vre_repository.depth import DepthRepresentation
from vre_repository.normals import NormalsRepresentation
from vre_repository.semantic_segmentation import SemanticRepresentation

def plot_one(data: MultiTaskItem, title: str, order: list[str] | None,
             name_to_task: dict[str, Representation], return_origs: bool = False) \
        -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """simple plot function: plot_one(reader[0][0], reader[0][1], None, reader.name_to_task)"""
    def vre_plot_fn(rgb_img: np.ndarray, x: tr.Tensor, task: Representation) -> np.ndarray:
        task.data = ReprOut(frames=rgb_img, output=MemoryData(x.cpu().detach().numpy()[None]), key=[0])
        try:
            res = task.make_images(task.data)[0]
        except Exception as e:
            logger.debug(f"Failed task '{task}': {task.data}")
            raise e
        return res
    name_to_task["rgb"].data = ReprOut(frames=None, output=MemoryData(data["rgb"].detach().numpy())[None], key=[0])
    rgb_img = name_to_task["rgb"].make_images(name_to_task["rgb"].data)
    img_data = {k: vre_plot_fn(rgb_img, v, name_to_task[k]) for k, v in data.items()}
    img_data = reorder_dict(img_data, order) if order is not None else img_data
    titles = [title if len(title) < 40 else f"{title[0:19]}..{title[-19:]}" for title in img_data]
    collage = collage_fn(list(img_data.values()), titles=titles, size_px=40)
    collage = image_add_title(collage, title, size_px=55, top_padding=110)
    return collage if return_origs is False else (collage, img_data)

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
                                  color_map=coco_color_map, disk_data_argmax=True)
m2f_mapillary = SemanticRepresentation("semantic_mask2former_mapillary_49189528_0", classes=mapillary_classes,
                                       color_map=mapillary_color_map, disk_data_argmax=True)
m2f_r50_mapillary = SemanticRepresentation("semantic_mask2former_mapillary_49189528_1", classes=mapillary_classes,
                                           color_map=mapillary_color_map, disk_data_argmax=True)
marigold = DepthRepresentation("depth_marigold", min_depth=0, max_depth=1)
normals_svd_marigold = NormalsRepresentation("normals_svd(depth_marigold)")

class SemanticMask2FormerMapillaryConvertedPaper(TaskMapper, SemanticRepresentation):
    def __init__(self, name: str, dependencies: list[SemanticRepresentation]):
        TaskMapper.__init__(self, name=name, n_channels=8, dependencies=dependencies)
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
        color_map = [[0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 255, 255],
                     [255, 0, 0], [0, 0, 255], [0, 255, 255], [127, 127, 63]]
        SemanticRepresentation.__init__(self, name, dependencies=dependencies, classes=list(self.mapping),
                                        color_map=color_map, disk_data_argmax=True)
        self.original_classes = dependencies[0].classes
        assert set(reduce(lambda x, y: x + y, self.mapping.values(), [])) == set(self.original_classes)

    @overrides
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        m2f_mapillary_converted = semantic_mapper(dep_data[0].argmax(-1), self.mapping, self.original_classes)
        return self.disk_to_memory_fmt(m2f_mapillary_converted)

class SemanticMask2FormerCOCOConverted(TaskMapper, SemanticRepresentation):
    def __init__(self, name: str, dependencies: list[Representation]):
        TaskMapper.__init__(self, name=name, n_channels=8, dependencies=dependencies)
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
        color_map = [[0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 255, 255],
                     [255, 0, 0], [0, 0, 255], [0, 255, 255], [127, 127, 63]]
        SemanticRepresentation.__init__(self, name, dependencies=dependencies, classes=list(self.mapping),
                                        color_map=color_map, disk_data_argmax=True)
        self.original_classes = dependencies[0].classes

    @overrides
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        m2f_mapillary_converted = semantic_mapper(dep_data[0].argmax(-1), self.mapping, self.original_classes)
        res = self.disk_to_memory_fmt(m2f_mapillary_converted)
        return res

class BinaryMapper(TaskMapper, NpIORepresentation):
    """
    Note for future self: this is never generic enough to be in VRE -- we'll keep it in this separate code only
    TaskMapper is the only high level interface that makes sense, so we should focus on keeping that generic and easy.
    """
    def __init__(self, name: str, dependencies: list[Representation], mapping: list[dict[str, list]],
                 mode: str, load_mode: str = "binary"):
        TaskMapper.__init__(self, name=name, dependencies=dependencies, n_channels=2)
        NpIORepresentation.__init__(self)
        assert mode in ("all_agree", "at_least_one", "majority"), (name, mode)
        assert load_mode in ("one_hot", "binary"), (name, load_mode)
        assert len(mapping[0]) == 2, (name, mapping)
        assert len(mapping) == len(dependencies), (name, len(mapping), len(dependencies))
        assert all(mapping[0].keys() == m.keys() for m in mapping), (name, [m.keys() for m in mapping])
        self.original_classes: list[list[str]] = [dep.classes for dep in dependencies]
        self.mapping = mapping
        self.mode = mode
        self.load_mode = load_mode
        self.classes = list(mapping[0].keys())
        self.n_classes = len(self.classes)
        self.color_map = [[0, 0, 0], [255, 255, 255]]
        self.output_dtype = "bool"

    @overrides
    def make_images(self, data: ReprOut) -> np.ndarray:
        x = data.output.argmax(-1) if self.load_mode == "one_hot" else (data.output > 0.5).astype(int)
        x = x[..., 0] if x.shape[-1] == 1 else x
        return colorize_semantic_segmentation(x, self.classes, self.color_map)

    @overrides
    def disk_to_memory_fmt(self, disk_data: DiskData) -> MemoryData:
        assert len(disk_data.shape) == 2 and disk_data.dtype == bool, f"{self.name}: {lo(disk_data)}"
        y = np.eye(2)[disk_data.astype(int)] if self.load_mode == "one_hot" else disk_data
        return MemoryData(y.astype(np.float32))

    @overrides
    def memory_to_disk_fmt(self, memory_data: MemoryData) -> DiskData:
        return memory_data.argmax(-1).astype(bool) if self.load_mode == "one_hot" else memory_data.astype(bool)

    @overrides
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        dep_data_argmaxed = [data.argmax(-1) for data in dep_data]
        dep_data_converted = [semantic_mapper(x, mapping, oc)
                              for x, mapping, oc in zip(dep_data_argmaxed, self.mapping, self.original_classes)]

        if self.mode == "all_agree":
            res_argmax = sum(dep_data_converted) == len(self.dependencies)
        elif self.mode == "at_least_one":
            res_argmax = sum(dep_data_converted) > 0
        else:
            res_argmax = sum(dep_data_converted) > len(dep_data_converted) // 2
        return self.disk_to_memory_fmt(res_argmax)

class BuildingsFromM2FDepth(BinaryMapper):
    def __init__(self, name: str, dependencies: list[Representation], buildings: BinaryMapper, mode: str,
                 load_mode: str = "binary"):
        assert len(dependencies) == 1, dependencies
        BinaryMapper.__init__(self, name=name, dependencies=buildings.dependencies,
                              mapping=buildings.mapping, mode=mode, load_mode=load_mode)
        self.dependencies = [*buildings.dependencies, dependencies[0]]
        self.classes = ["others", name]

    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        buildings = super().merge_fn(dep_data[0:-1])
        depth = dep_data[-1] if len(dep_data[-1].shape) == 2 else dep_data[-1][..., 0]
        thr = 0.3 # np.percentile(depth.numpy(), 0.8)
        buildings_depth = buildings * (depth <= thr)
        return self.disk_to_memory_fmt(buildings_depth.astype(bool))

class SemanticMedian(TaskMapper, SemanticRepresentation):
    def __init__(self, name: str, deps: list[TaskMapper | SemanticRepresentation]):
        assert all(dep.n_channels == deps[0].n_channels for dep in deps), [(dep.name, dep.n_channels) for dep in deps]
        TaskMapper.__init__(self, name, n_channels=deps[0].n_channels, dependencies=deps)
        SemanticRepresentation.__init__(self, name, dependencies=deps, classes=deps[0].classes,
                                        color_map=deps[0].color_map, disk_data_argmax=True)

    @overrides
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        return MemoryData(np.eye(self.n_classes)[sum(dep_data).argmax(-1)].astype(np.float32))

class SafeLandingAreas(BinaryMapper):
    def __init__(self, name: str, depth: DepthRepresentation, camera_normals: NormalsRepresentation,
                 include_semantics: bool, original_classes: tuple[list[str], list[str]] | None = None,
                 semantics: list[SemanticRepresentation] | None = None, load_mode: str = "binary"):
        dependencies = [depth, camera_normals]
        if include_semantics:
            assert len(original_classes) == 3
            assert len(semantics) == 3
            dependencies = [*dependencies, *semantics]
        TaskMapper.__init__(self, name, dependencies=dependencies, n_channels=2)
        self.color_map = [[255, 0, 0], [0, 255, 0]]
        self.original_classes = original_classes
        self.classes = ["unsafe-landing", "safe-landing"]
        self.n_classes = len(self.classes)
        self.semantics = semantics
        self.load_mode = load_mode
        self.include_semantics = include_semantics
        self.output_dtype = "bool"

        safe_coco_classes = ["grass-merged", "dirt-merged", "sand", "gravel", "flower", "playingfield", "snow", "road",
                             "platform", "railroad", "pavement-merged", "mountain-merged", "roof", "tree-merged",
                             "rock-merged"]
        safe_mapillary_classes = ["Terrain", "Sand", "Snow", "Road", "Lane Marking - General", "Sidewalk", "Bridge",
                                  "Pothole", "Catch Basin", "Tunnel", "Parking", "Service Lane", "Pedestrian Area",
                                  "Lane Marking - Crosswalk", "On Rails", "Bike Lane", "Crosswalk - Plain", "Mountain",
                                  "Vegetation"]

        self.safe_coco_ix, self.safe_mapillary_ix = None, None
        if include_semantics:
            assert all(X := [c in original_classes[1] for c in safe_coco_classes]), list(zip(coco_classes, X))
            assert all(X := [c in original_classes[0] for c in safe_mapillary_classes]), \
                list(zip(safe_mapillary_classes, X))
            self.safe_coco_ix = [coco_classes.index(ix) for ix in safe_coco_classes]
            self.safe_mapillary_ix = [mapillary_classes.index(ix) for ix in safe_mapillary_classes]

    @overrides
    def merge_fn(self, dep_data: list[MemoryData]) -> MemoryData:
        depth, normals = dep_data[0] if len(dep_data[0].shape) == 2 else dep_data[0][..., 0], dep_data[1]
        v1, v2, v3 = normals.transpose(2, 0, 1)
        where_safe = (v2 > 0.8) * ((v1 + v3) < 1.2) * (depth <= 0.9)
        if self.include_semantics:
            conv1 = np.isin(dep_data[2].argmax(-1), self.safe_mapillary_ix).astype(int)
            conv2 = np.isin(dep_data[3].argmax(-1), self.safe_coco_ix).astype(int)
            conv3 = np.isin(dep_data[4].argmax(-1), self.safe_mapillary_ix).astype(int)
            sema_safe = (conv1 + conv2 + conv3) >= 2
            where_safe = sema_safe * where_safe
        return self.disk_to_memory_fmt(where_safe)

def get_new_semantic_mapped_tasks(tasks_subset: list[str] | None = None,
                                  include_semantic_output: bool = True) -> dict[str, TaskMapper]:
    """The exported function for VRE!"""
    buildings_mapping = [
        {
            "others": [x for x in mapillary_classes if x not in
                       (cls := ["Building", "Utility Pole", "Pole", "Fence", "Wall"])],
            "buildings": cls,
        },
        {
            "others": [x for x in coco_classes if x not in (cls := ["building-other-merged", "house", "roof"])],
            "buildings": cls,
        },
        {
            "others": [x for x in mapillary_classes if x not in
                       (cls := ["Building", "Utility Pole", "Pole", "Fence", "Wall"])],
            "buildings": cls,
        },
    ]

    sky_and_water_mapping = [
        {
            "others": [c for c in mapillary_classes if c not in (cls := ["Sky", "Water"])],
            "sky-and-water": cls,
        },
        {
            "others": [c for c in coco_classes if c not in
                       (cls := ["sky-other-merged", "water-other", "sea", "river"])],
            "sky-and-water": cls,
        },
        {
            "others": [c for c in mapillary_classes if c not in (cls := ["Sky", "Water"])],
            "sky-and-water": cls,
        },
    ]

    transportation_mapping = [
        {
            "others": [c for c in mapillary_classes if c not in
                       (cls := ["Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Rail Track", "Road",
                                "Service Lane", "Sidewalk", "Bridge", "Tunnel", "Bicyclist", "Motorcyclist",
                                "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General", "Traffic Light",
                                "Traffic Sign (Back)", "Traffic Sign (Front)", "Bicycle", "Boat", "Bus", "Car",
                                "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck",
                                "Wheeled Slow", "Car Mount", "Ego Vehicle"])],
            "transportation": cls,
        },
        {
            "others": [c for c in coco_classes if c not in
                       (cls := ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                                "road", "railroad", "pavement-merged"])],
            "transportation": cls,
        },
        {
            "others": [c for c in mapillary_classes if c not in
                       (cls := ["Bike Lane", "Crosswalk - Plain", "Curb Cut", "Parking", "Rail Track", "Road",
                                "Service Lane", "Sidewalk", "Bridge", "Tunnel", "Bicyclist", "Motorcyclist",
                                "Other Rider", "Lane Marking - Crosswalk", "Lane Marking - General", "Traffic Light",
                                "Traffic Sign (Back)", "Traffic Sign (Front)", "Bicycle", "Boat", "Bus", "Car",
                                "Caravan", "Motorcycle", "On Rails", "Other Vehicle", "Trailer", "Truck",
                                "Wheeled Slow", "Car Mount", "Ego Vehicle"])],
            "transportation": cls,
        },
    ]

    containing_mapping = [
        {
            "contained": [c for c in mapillary_classes if c not in
                          (cls := ["Terrain", "Sand", "Mountain", "Road", "Sidewalk", "Pedestrian Area", "Rail Track",
                                   "Parking", "Service Lane", "Bridge", "Water", "Curb", "Fence", "Wall",
                                   "Guard Rail", "Barrier", "Curb Cut", "Snow"])],
            "containing": cls,
        },
        {
            "contained": [c for c in coco_classes if c not in
                          (cls := ["floor-wood", "floor-other-merged", "pavement-merged", "mountain-merged", "platform",
                                   "sand", "road", "sea", "river", "railroad", "grass-merged", "snow", "stairs",
                                   "tent"])],
            "containing": cls,
        },
        {
            "contained": [c for c in mapillary_classes if c not in
                          (cls := ["Terrain", "Sand", "Mountain", "Road", "Sidewalk", "Pedestrian Area", "Rail Track",
                                   "Parking", "Service Lane", "Bridge", "Water", "Curb", "Fence", "Wall",
                                   "Guard Rail", "Barrier", "Curb Cut", "Snow"])],
            "containing": cls,
        },
    ]

    vegetation_mapping = [
        {
            "others": [x for x in mapillary_classes if x not in
                       (cls := ["Mountain", "Sand", "Snow", "Terrain", "Vegetation"])],
            "vegetation": cls,
        },
        {
            "others": [x for x in coco_classes if x not in
                       (cls := ["tree-merged", "grass-merged", "dirt-merged", "flower", "potted plant", "river",
                                "sea", "water-other", "mountain-merged", "rock-merged"])],
            "vegetation": cls,
        },
        {
            "others": [x for x in mapillary_classes if x not in
                       (cls := ["Mountain", "Sand", "Snow", "Terrain", "Vegetation"])],
            "vegetation": cls,
        },
    ]

    available_tasks: list[TaskMapper] = [
        m2f_swin_mapillary_converted := SemanticMask2FormerMapillaryConvertedPaper(
            "semantic_mask2former_swin_mapillary_converted", [m2f_mapillary]),
        m2f_r50_mapillary_converted := SemanticMask2FormerMapillaryConvertedPaper(
            "semantic_mask2former_r50_mapillary_converted", [m2f_r50_mapillary]),
        m2f_swin_coco_converted := SemanticMask2FormerCOCOConverted(
            "semantic_mask2former_swin_coco_converted", [m2f_coco]),
        buildings := BinaryMapper("buildings", [m2f_mapillary, m2f_coco, m2f_r50_mapillary],
                                  buildings_mapping, mode="majority"),
        BinaryMapper("sky-and-water", [m2f_mapillary, m2f_coco, m2f_r50_mapillary],
                     sky_and_water_mapping, mode="majority"),
        BinaryMapper("transportation", [m2f_mapillary, m2f_coco, m2f_r50_mapillary],
                     transportation_mapping, mode="majority"),
        BinaryMapper("containing", [m2f_mapillary, m2f_coco, m2f_r50_mapillary], containing_mapping, mode="majority"),
        BinaryMapper("vegetation", [m2f_mapillary, m2f_coco, m2f_r50_mapillary], vegetation_mapping, mode="majority"),
        BuildingsFromM2FDepth("buildings(nearby)", [marigold], buildings, mode="majority"),
        SafeLandingAreas("safe-landing-no-sseg", marigold, normals_svd_marigold, include_semantics=False),
        SafeLandingAreas("safe-landing-semantics", marigold, normals_svd_marigold, include_semantics=True,
                         original_classes=[mapillary_classes, coco_classes, mapillary_classes],
                         semantics=[m2f_mapillary, m2f_coco, m2f_r50_mapillary]),
    ]
    if include_semantic_output:
        available_tasks.append(SemanticMedian("semantic_output", [m2f_swin_mapillary_converted,
                                                                  m2f_r50_mapillary_converted,
                                                                  m2f_swin_coco_converted]))
    if tasks_subset is None:
        return {t.name: t for t in available_tasks}
    return {t.name: t for t in available_tasks if t.name in tasks_subset}

if __name__ == "__main__":
    cfg_path = Path.cwd() / "cfg.yaml"
    data_path = Path.cwd() / "data"
    vre_dir = data_path

    task_names = ["rgb", "depth_marigold", "normals_svd(depth_marigold)",
                  "semantic_mask2former_coco_47429163_0", "semantic_mask2former_mapillary_49189528_0",
                  "semantic_mask2former_mapillary_49189528_1"]
    order = ["rgb", "semantic_mask2former_mapillary_49189528_0", "semantic_mask2former_coco_47429163_0",
             "depth_marigold", "normals_svd(depth_marigold)"]

    repr_types = get_vre_repository()
    task_types = {r.name: r for r in build_representations_from_cfg(cfg_path, repr_types) if r.name in task_names}
    reader = MultiTaskDataset(vre_dir, task_names=task_names, task_types=task_types,
                              handle_missing_data="fill_nan", normalization=None,
                              cache_task_stats=True, batch_size_stats=100)
    orig_task_names = list(reader.task_types.keys())

    new_tasks = get_new_semantic_mapped_tasks(include_semantic_output=False) # TODO: depth=2 not implemented in reader
    for task_name in reader.task_names:
        if task_name not in orig_task_names:
            reader.remove_task(task_name)
    for new_task in new_tasks.values():
        reader.add_task(new_task, overwrite=True)

    # Note: depth=2 works well in VRE, just not in MultiTaskDataset (need to disable the assert for depth <= 1 tho)
    # from vre import VRE, FakeVideo
    # from vre.representations.build_representations import _add_external_representations_dict
    # reprs = _add_external_representations_dict(list(task_types.values()),
    #                                            get_new_semantic_mapped_tasks(include_semantic_output=True), {}, {}, {})
    # video= FakeVideo(np.array(list(map(lambda x: np.load(x)["arr_0"], (data_path/"rgb/npz").iterdir()))),
    #                  fps=1, frames=[5,8,22])
    # vre = VRE(video, reprs)
    # vre._compute_one_representation_batch(vre["semantic_output"], [5,8,22], Path.cwd())
    # vre.to_graphviz().render("graph", format="png", cleanup=True)

    print("== Random loaded item ==")
    ixs = np.random.permutation(range(len(reader))).tolist()
    ixs = ["8.npz"]
    for ix in ixs:
        data, name = reader[ix]
        pprint(data)
        res, origs = plot_one(data, title=name, order=order, name_to_task=reader.name_to_task, return_origs=True)
        print(f"{name} -- {res.shape}")
        image_write(res, f"collage_{name[0:-4]}.png")
        for k, v in origs.items():
            image_write(v, f"collage_{name[0:-4]}_{k}.png")
        break
