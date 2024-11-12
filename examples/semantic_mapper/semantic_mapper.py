#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from pprint import pprint
from loggez import loggez_logger as logger
from omegaconf import OmegaConf
from overrides import overrides
import torch as tr
import numpy as np
import matplotlib.pyplot as plt

from vre import VRE
from vre.utils import (semantic_mapper, colorize_semantic_segmentation, image_add_title, image_write, collage_fn,
                       FFmpegVideo)
from vre.readers import MultiTaskDataset
from vre.representations import TaskMapper, NpIORepresentation, ReprOut, build_representations_from_cfg, Representation

sys.path.append(Path(__file__).parent.__str__())
from dronescapes_representations import dronescapes_task_types, coco_classes, mapillary_classes

INCLUDE_SEMANTICS_ORIGINAL = True
VRE_N_FRAMES = 5

### Some utils function for plotting and running VRE if needed (on a video)

def vre_plot_fn(x: tr.Tensor, node: Representation) -> np.ndarray:
    x = x.cpu().numpy()
    node.data = ReprOut(None, x[None], [0])
    res = node.make_images()[0]
    return res

def reorder_dict(data: dict[str, "Any"], keys: list[str]) -> dict[str, "Any"]:
    assert (diff := set(keys).difference(data.keys())) == set(),diff
    for k in keys[::-1]:
        data = {k: data[k], **{k: v for k, v in data.items() if data != k}}
    return data

def plot_one(data: "MultiTaskItem", title: str, order: list[str],
             name_to_task: dict[str, Representation]) -> np.ndarray:
    print(title)
    img_data = {k: vre_plot_fn(v, name_to_task[k]) for k, v in data.items()}
    img_data = reorder_dict(img_data, order)
    titles = [title if len(title) < 40 else f"{title[0:19]}..{title[-19:]}" for title in img_data]
    collage = collage_fn(list(img_data.values()), titles=titles, size_px=40)
    collage = image_add_title(collage, title, size_px=55, top_padding=110)
    return collage

def run_vre(video_path: Path, vre_path: Path | None, frames: int | list[int]=VRE_N_FRAMES):
    vre_path = vre_path or Path.cwd() / f"data_{video_path.name}"
    frames = frames if isinstance(frames, list) else np.random.choice(len(video), size=frames, replace=True).tolist()
    os.environ["VRE_DEVICE"] = ("cuda" if tr.cuda.is_available() else "cpu")
    representations = build_representations_from_cfg(OmegaConf.load(Path(__file__).parent / "cfg.yaml"))
    if vre_path.exists() and \
            all(((D := vre_path/repr/"npz").exists()) and
                set(list(map(lambda x: int(x.stem), D.iterdir()))) == set(frames) for repr in representations):
        logger.info(f"{vre_path} already computed, using as-is w/o calling VRE again. Delete it if you want again")
        return vre_path
    video = FFmpegVideo(video_path)
    vre = VRE(video, representations)
    vre.run(vre_path, frames=frames, n_threads_data_storer=2)
    return vre_path

### Representations only below

class BinaryMapper(TaskMapper, NpIORepresentation):
    """
    Note for future self: this is never generic enough to be in VRE -- we'll keep it in this separate code only
    TaskMapper is the only high level interface that makes sense, so we should focus on keeping that generic and easy.
    """
    def __init__(self, name: str, dependencies: list, mapping: list[dict[str, list]],
                 color_map: list[tuple[int, int, int]], mode: str = "all_agree"):
        TaskMapper.__init__(self, name=name, dependencies=dependencies, n_channels=2)
        NpIORepresentation.__init__(self)
        assert mode in ("all_agree", "at_least_one"), mode
        assert len(mapping[0]) == len(color_map), (len(mapping[0]), len(color_map))
        assert len(mapping[0]) == 2, mapping
        assert len(mapping) == len(dependencies), (len(mapping), len(dependencies))
        assert all(mapping[0].keys() == m.keys() for m in mapping), [m.keys() for m in mapping]
        self.original_classes = [dep.classes for dep in dependencies]
        self.mapping = mapping
        self.classes = list(mapping[0].keys())
        self.n_classes = len(self.classes)
        self.mode = mode
        self.color_map = [[255, 255, 255], [0, 0, 0]]
        self.output_dtype = "bool"

    @overrides
    def make_images(self) -> np.ndarray:
        res = [colorize_semantic_segmentation(item.argmax(-1).astype(int), self.classes, color_map=self.color_map,
                                              original_rgb=None, font_size_scale=2) for item in self.data.output]
        return np.array(res)

    @overrides
    def from_disk_fmt(self, disk_data: np.ndarray) -> np.ndarray:
        return np.eye(2)[disk_data.astype(int)].astype(np.float32)

    @overrides
    def to_disk_fmt(self, memory_data: np.ndarray) -> np.ndarray:
        return memory_data.argmax(-1).astype(bool)

    @overrides
    def merge_fn(self, dep_data: list[np.ndarray]) -> np.ndarray:
        dep_data_converted = [semantic_mapper(x.argmax(-1), mapping, oc)
                              for x, mapping, oc in zip(dep_data, self.mapping, self.original_classes)]
        res_argmax = sum(dep_data_converted) > (0 if self.mode == "all_agree" else 1)
        return np.eye(2)[res_argmax.astype(int)].astype(np.float32)

class BuildingsFromM2FDepth(TaskMapper, NpIORepresentation):
    def __init__(self, name: str, original_classes: tuple[list[str], list[str]]):
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

        dependencies = [dronescapes_task_types["semantic_mask2former_mapillary_49189528_0"],
                        dronescapes_task_types["semantic_mask2former_coco_47429163_0"],
                        dronescapes_task_types["depth_marigold"]]
        TaskMapper.__init__(self, name=name, dependencies=dependencies, n_channels=2)
        NpIORepresentation.__init__(self)
        self.color_map = [[255, 255, 255], [0, 0, 0]]
        self.original_classes = original_classes
        self.mapping = buildings_mapping
        self.classes = list(buildings_mapping[0].keys())
        self.n_classes = len(self.classes)

    @overrides
    def make_images(self) -> np.ndarray:
        res = [colorize_semantic_segmentation(item.argmax(-1).astype(int), self.classes, self.color_map,
                                              original_rgb=None, font_size_scale=2) for item in self.data.output]
        return np.array(res)

    def merge_fn(self, dep_data: list[np.ndarray]) -> np.ndarray:
        m2f_mapillary, m2f_coco = dep_data[0].argmax(-1), dep_data[1].argmax(-1)
        depth = dep_data[2].squeeze()
        m2f_mapillary_converted = semantic_mapper(m2f_mapillary, self.mapping[0], self.original_classes[0])
        m2f_coco_converted = semantic_mapper(m2f_coco, self.mapping[1], self.original_classes[1])
        thr = 0.3 # np.percentile(depth.numpy(), 0.8)
        combined = (m2f_mapillary_converted + m2f_coco_converted + (depth > thr)) != 0
        return np.eye(2)[combined.astype(int)]

class SafeLandingAreas(TaskMapper, NpIORepresentation):
    def __init__(self, name: str, original_classes: tuple[list[str], list[str]], include_semantics: bool = False,
                 sky_water: BinaryMapper | None = None):
        self.include_semantics = include_semantics
        dependencies = [dronescapes_task_types["semantic_mask2former_mapillary_49189528_0"],
                        dronescapes_task_types["semantic_mask2former_coco_47429163_0"],
                        dronescapes_task_types["depth_marigold"],
                        dronescapes_task_types["normals_svd(depth_marigold)"]]
        TaskMapper.__init__(self, name, dependencies=dependencies, n_channels=2)
        NpIORepresentation.__init__(self)
        self.color_map = [[0, 255, 0], [255, 0, 0]]
        self.original_classes = original_classes
        self.classes = ["safe-landing", "unsafe-landing"]
        self.n_classes = len(self.classes)
        self.sky_water = sky_water

    @overrides
    def make_images(self) -> np.ndarray:
        res = [colorize_semantic_segmentation(item.argmax(-1).astype(int), self.classes, color_map=self.color_map,
                                              original_rgb=None, font_size_scale=2) for item in self.data.output]
        return np.array(res)

    @overrides
    def merge_fn(self, dep_data: list[np.ndarray]) -> np.ndarray:
        normals, depth = dep_data[3], dep_data[2].squeeze()
        v1, v2, v3 = normals.transpose(2, 0, 1)
        where_safe = (v2 > 0.8) * ((v1 + v3) < 1.2)
        if self.include_semantics:
            sw = self.sky_water.merge_fn(dep_data).argmax(-1)
            where_safe = (where_safe * sw * (depth < 0.9)).astype(bool)
        return np.eye(2)[(~where_safe).astype(int)]

def get_new_dronescapes_tasks(tasks_subset: list[str] | None = None) -> dict[str, TaskMapper]:
    color_map_binary = [[255, 255, 255], [0, 0, 0]]
    sem_mapillary, sem_coco = [dronescapes_task_types["semantic_mask2former_mapillary_49189528_0"],
                               dronescapes_task_types["semantic_mask2former_coco_47429163_0"]]
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

    available_tasks: list[TaskMapper] = [
        BinaryMapper("buildings", [sem_mapillary, sem_coco], buildings_mapping, color_map=color_map_binary),
        BinaryMapper("living-vs-non-living", [sem_mapillary, sem_coco], living_mapping, color_map=color_map_binary),
        sky_water := BinaryMapper("sky-and-water", [sem_mapillary, sem_coco], sky_and_water_mapping,
                                  color_map=color_map_binary, mode="at_least_one"),
        BinaryMapper("transportation", [sem_mapillary, sem_coco], transportation_mapping,
                     color_map=color_map_binary, mode="at_least_one"),
        BinaryMapper("containing", [sem_mapillary, sem_coco], containing_mapping, color_map=color_map_binary),
        BuildingsFromM2FDepth("buildings(nearby)", [mapillary_classes, coco_classes]),
        SafeLandingAreas("safe-landing-no-sseg", [mapillary_classes, coco_classes]),
        SafeLandingAreas("safe-landing-semantics", [mapillary_classes, coco_classes],
                         include_semantics=True, sky_water=sky_water),
    ]
    if tasks_subset is None:
        return {t.name: t for t in available_tasks}
    return {t.name: t for t in available_tasks if t.name in tasks_subset}

def add_tasks_to_reader(reader: MultiTaskDataset, orig_task_names: list[str]):
    for task_name in reader.task_names:
        if task_name not in orig_task_names:
            reader.remove_task(task_name)
    for new_task in get_new_dronescapes_tasks().values():
        reader.add_task(new_task, overwrite=True)

def main():
    task_names = ["rgb", "depth_marigold", "normals_svd(depth_marigold)", "opticalflow_rife",
                  "semantic_mask2former_swin_mapillary_converted", "semantic_mask2former_swin_coco_converted"]
    order = ["rgb", "depth_marigold", "normals_svd(depth_marigold)"]
    if INCLUDE_SEMANTICS_ORIGINAL:
        task_names.extend(["semantic_mask2former_coco_47429163_0", "semantic_mask2former_mapillary_49189528_0"])
        order = ["rgb", "semantic_mask2former_mapillary_49189528_0", "semantic_mask2former_coco_47429163_0",
                 "depth_marigold", "normals_svd(depth_marigold)"]
    data_path = Path(sys.argv[1])
    if data_path.suffix == ".mp4":
        logger.info(f"{data_path} is a video. Running VRE first to get the raw representations")
        data_path = run_vre(data_path, Path.cwd() / f"data_{data_path.name}")

    reader = MultiTaskDataset(data_path, task_names=task_names,
                              task_types=dronescapes_task_types, handle_missing_data="fill_nan",
                              normalization="min_max", cache_task_stats=True, batch_size_stats=100)
    print(reader)
    print("== Shapes ==")
    pprint(reader.data_shape)

    orig_task_names = list(reader.task_types.keys())
    add_tasks_to_reader(reader, orig_task_names)

    print("== Random loaded item ==")
    ixs = np.random.permutation(range(len(reader))).tolist()
    data, name, _ = reader[ixs[0]] # get a random item
    print(name)
    img_data = {}
    for k, v in data.items():
        assert v is not None, k
        reader.name_to_task[k].data = ReprOut(None, v.numpy()[None], ixs[0:1], None)
        img_data[k] = reader.name_to_task[k].make_images()[0]
    img_data = reorder_dict(img_data, order)
    titles = [title if len(title) < 40 else f"{title[0:19]}..{title[-19:]}" for title in img_data]
    collage = collage_fn(list(img_data.values()), titles=titles, size_px=40)
    collage = image_add_title(collage, name, size_px=55, top_padding=110)
    plt.figure(figsize=(20, 10))
    plt.imshow(collage)
    image_write(collage, out_path := f"collage_{name[0:-4]}.png")
    logger.info(f"Stored at '{out_path}'")

if __name__ == "__main__":
    main()
