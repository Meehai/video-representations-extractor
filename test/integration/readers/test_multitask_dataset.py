from pathlib import Path
from tempfile import TemporaryDirectory
from vre.readers import MultiTaskDataset
from vre.representations.cv_representations import RGBRepresentation, DepthRepresentation, SemanticRepresentation
from vre.representations import NormedRepresentationMixin
from torch.utils.data import DataLoader
import numpy as np
import torch as tr
import pytest

color_map_8classes = [[0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 255, 255],
                    [255, 0, 0], [0, 0, 255], [0, 255, 255], [127, 127, 63]]
fake_dronescapes_task_types = {
    "rgb": RGBRepresentation("rgb"),
    "depth_sfm_manual202204": DepthRepresentation("depth_sfm_manual202204", min_depth=0, max_depth=300),
    "semantic_segprop8": SemanticRepresentation("semantic_segprop8", classes=8, color_map=color_map_8classes),
}

def _dataset_path() -> Path:
    (temp_dir := Path(TemporaryDirectory().name)).mkdir(exist_ok=False)
    task_names = ("rgb", "semantic_segprop8", "depth_sfm_manual202204")

    N, H, W = 33, 16, 25
    n_classes = fake_dronescapes_task_types["semantic_segprop8"].n_classes
    min_depth, max_depth = 0, 300
    data = {
        "rgb": np.random.randint(0, 255, size=(N, H, W, 3)).astype(np.uint8),
        "semantic_segprop8": np.random.randint(0, n_classes, size=(N, H, W)).astype(np.uint16),
        "depth_sfm_manual202204": np.random.random(size=(N, H, W, 1)).astype(np.float32) \
            * (max_depth - min_depth) + min_depth,
    }

    for task in task_names:
        Path(temp_dir / task).mkdir(exist_ok=False, parents=False)
        for i, item in enumerate(data[task]):
            np.savez(temp_dir / task / f"{i}.npz", item)
    return temp_dir

@pytest.fixture
def dataset_path() -> Path: return _dataset_path()

def test_MultiTaskDataset_ctor_shapes_and_types(dataset_path: Path):
    N, H, W = 33, 16, 25
    dataset = MultiTaskDataset(dataset_path, task_names=list(fake_dronescapes_task_types.keys()),
                               task_types=fake_dronescapes_task_types, normalization=None)
    assert len(dataset) == N
    assert dataset.task_names == ["depth_sfm_manual202204", "rgb", "semantic_segprop8"], dataset.task_names # inferred
    expected_shapes = {"rgb": (H, W, 3), "semantic_segprop8": (H, W, 8), "depth_sfm_manual202204": (H, W, 1)}
    assert (A := dataset.data_shape) == (B := expected_shapes), f"\n- {A} vs\n- {B}"
    x, _, __ = dataset[0]

    assert x["rgb"].min() == 0 and x["rgb"].max() >= 240 and x["rgb"].max() <= 255, x["rgb"]
    assert x["rgb"].dtype == tr.uint8, x["rgb"]
    assert x["rgb"].shape == (H, W, 3)
    assert x["depth_sfm_manual202204"].min() <= 10 and x["depth_sfm_manual202204"].max() >= 290, \
        x["depth_sfm_manual202204"]
    assert x["depth_sfm_manual202204"].dtype == tr.float32, x["depth_sfm_manual202204"]
    assert x["depth_sfm_manual202204"].shape == (H, W, 1)
    assert x["semantic_segprop8"].unique().tolist() == [0, 1], x["semantic_segprop8"]
    assert x["semantic_segprop8"].dtype == tr.float32 # overwrite in dronescapes representation
    assert x["semantic_segprop8"].shape == (H, W, 8)

@pytest.mark.parametrize("normalization", ["standardization", "min_max"])
def test_MultiTaskDataset_normalization(dataset_path: Path, normalization: str):
    dataset = MultiTaskDataset(dataset_path, task_names=list(fake_dronescapes_task_types.keys()),
                               task_types=fake_dronescapes_task_types, normalization=normalization)
    x, _, __ = dataset[0]
    for task in dataset.task_names:
        assert x[task].dtype == tr.float32, x[task].dtype
        if normalization == "min_max":
            assert x[task].min() >= 0 and x[task].max() <= 1, x[task]
        else:
            if not isinstance(dataset.name_to_task[task], NormedRepresentationMixin):
                continue
            MAX_MEAN_DIFF, MAX_STD_DIFF, MIN_STD, MAX_STD = 0.2, 0.2, -5, 5 # TODO: why so big??
            assert x[task].min() >= MIN_STD and x[task].max() <= MAX_STD, x[task]
            assert x[task].mean().abs() < MAX_MEAN_DIFF and (x[task].std() - 1).abs() < MAX_STD_DIFF, \
                (task, x[task].mean(), x[task].std())

def test_MultiTaskDataset_getitem(dataset_path):
    reader = MultiTaskDataset(dataset_path, task_names=list(fake_dronescapes_task_types.keys()),
                              task_types=fake_dronescapes_task_types, normalization=None)

    rand_ix = np.random.randint(len(reader))
    data, _, repr_names = reader[rand_ix] # get a random single data point
    assert all(len(x.shape) == 3 for x in data.values()), data
    assert set(repr_names) == set(reader.task_names)

    data, _, repr_names = reader[rand_ix: min(len(reader), rand_ix + 5)] # get a random batch
    assert all(len(x.shape) == 4 for x in data.values()), data
    assert set(repr_names) == set(reader.task_names)

    loader = DataLoader(reader, collate_fn=reader.collate_fn, batch_size=5, shuffle=True)
    data, _, repr_names = next(iter(loader)) # get a random batch using torch DataLoader
    assert all(len(x.shape) == 4 for x in data.values()), data
    assert set(repr_names) == set(reader.task_names)

if __name__ == "__main__":
    test_MultiTaskDataset_normalization(_dataset_path(), "min_max")
